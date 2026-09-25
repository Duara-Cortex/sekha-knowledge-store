package consolidation_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/consolidation"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/recall"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/security"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

func createTestStore(t *testing.T, dbPath string) *store.SQLiteStore {
	_ = os.Remove(dbPath)
	_ = os.Remove(dbPath + "-wal")
	_ = os.Remove(dbPath + "-shm")

	s, err := store.NewSQLiteStore(dbPath)
	if err != nil {
		t.Fatalf("failed to initialise test SQLite store: %v", err)
	}

	t.Cleanup(func() {
		_ = s.Close()
		_ = os.Remove(dbPath)
		_ = os.Remove(dbPath + "-wal")
		_ = os.Remove(dbPath + "-shm")
	})

	return s
}

func TestExtractorSalienceAndCausalRelations(t *testing.T) {
	extractor := consolidation.NewExtractor(64)

	trace := model.EpisodicTrace{
		ID:        "trace-unit-01",
		SessionID: "session-abc-123",
		TaskGoal:  "Deploy and configure reverse proxy on edge node",
		Outcome:   model.OutcomeSuccess,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex:   1,
				Thought:     "Check port availability on target node",
				Action:      "netstat -tuln",
				Observation: "Port 8084 is free",
				Status:      "success",
			},
			{
				StepIndex:   2,
				Thought:     "Initialise reverse proxy service configuration",
				Action:      "systemctl restart proxy",
				Observation: "Service active",
				Status:      "success",
			},
		},
		SensoryContext: []model.SensoryItem{
			{
				Text:     "Edge cluster telemetry indicates normal thermals",
				Salience: 0.85,
			},
		},
	}

	res := extractor.Extract(trace)

	if res.Outcome != model.OutcomeSuccess {
		t.Errorf("expected outcome success, got %s", res.Outcome)
	}
	if res.Salience != 1.0 {
		t.Errorf("expected salience 1.0, got %f", res.Salience)
	}
	if len(res.Entities) == 0 {
		t.Fatalf("expected extracted entities, got 0")
	}
	if len(res.Edges) == 0 {
		t.Fatalf("expected extracted causal edges, got 0")
	}

	// Verify primary task goal was extracted
	var foundGoal bool
	for _, e := range res.Entities {
		if e.EntityType == "task_goal" {
			foundGoal = true
			if len(e.Embedding) != 64 {
				t.Errorf("expected 64-D embedding, got %d", len(e.Embedding))
			}
		}
	}
	if !foundGoal {
		t.Errorf("task_goal entity was not extracted")
	}
}

func TestGraphFusionAndEntityDeduplication(t *testing.T) {
	s := createTestStore(t, "test_fusion.db")
	cfg := model.DefaultDecayConfig()
	engine := consolidation.NewEngine(s, cfg)
	ctx := context.Background()

	// 1. Ingest first trace containing recurrent entity "reverse proxy"
	trace1 := model.ConsolidateRequest{
		TraceID:     "trace-01",
		SessionID:   "sess-01",
		TaskGoal:    "Setup reverse proxy on Node 1",
		Outcome:     model.OutcomeSuccess,
		Synchronous: true,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex: 1,
				Thought:   "Configuring proxy service route",
				Action:    "configure_proxy",
				Status:    "success",
			},
		},
	}

	resp1, err := engine.IngestTrace(ctx, trace1)
	if err != nil {
		t.Fatalf("failed ingesting trace 1: %v", err)
	}
	if resp1.Status != "consolidated" {
		t.Errorf("expected trace 1 consolidated, got %s", resp1.Status)
	}

	stats1, err := engine.GetStats(ctx)
	if err != nil {
		t.Fatalf("failed getting stats: %v", err)
	}
	initialActiveNodes := stats1.ActiveNodes

	// 2. Ingest second trace referencing identical goal and concepts
	trace2 := model.ConsolidateRequest{
		TraceID:     "trace-02",
		SessionID:   "sess-02",
		TaskGoal:    "Setup reverse proxy on Node 1",
		Outcome:     model.OutcomeSuccess,
		Synchronous: true,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex: 1,
				Thought:   "Verify reverse proxy status on Node 1",
				Action:    "check_status",
				Status:    "success",
			},
		},
	}

	resp2, err := engine.IngestTrace(ctx, trace2)
	if err != nil {
		t.Fatalf("failed ingesting trace 2: %v", err)
	}
	if resp2.NodesFused == 0 {
		t.Errorf("expected existing nodes to be fused and deduplicated")
	}

	// Verify that the task_goal node was deduplicated rather than duplicated
	node, err := s.FindMatchingNode(ctx, "Setup reverse proxy on Node 1", "task_goal")
	if err != nil || node == nil {
		t.Fatalf("expected to find deduplicated task goal node: %v", err)
	}
	if node.AccessCount <= 1 {
		t.Errorf("expected access count to increase for deduplicated node, got %d", node.AccessCount)
	}

	stats2, err := engine.GetStats(ctx)
	if err != nil {
		t.Fatalf("failed getting stats 2: %v", err)
	}
	// Active nodes should grow only by unique step actions, not duplicate goal
	if stats2.ActiveNodes >= initialActiveNodes*2 {
		t.Errorf("expected deduplication to prevent 2x node expansion (was %d, now %d)", initialActiveNodes, stats2.ActiveNodes)
	}
}

func TestMathematicalRecencyDecayAndSoftArchival(t *testing.T) {
	s := createTestStore(t, "test_decay.db")
	ctx := context.Background()

	// Configure decay with short half-life and grace period for deterministic testing
	cfg := model.DecayConfig{
		DecayHalfLife:         24 * time.Hour,
		PruneThreshold:        0.15,
		EdgePruneThreshold:    0.05,
		InactivityGracePeriod: 48 * time.Hour,
		HebbianLearningRate:   0.15,
		MaxEdgeWeight:         5.0,
		BatchSize:             100,
	}
	engine := consolidation.NewEngine(s, cfg)

	now := time.Now().UTC()

	// Ingest a transient trace that will receive no subsequent reinforcement
	trace := model.ConsolidateRequest{
		TraceID:     "transient-trace-01",
		SessionID:   "transient-sess",
		TaskGoal:    "Temporary speculative thought experiment",
		Outcome:     model.OutcomeFailure,
		Synchronous: true,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex: 1,
				Thought:   "Ephemeral debug attempt that failed",
				Action:    "speculative_eval",
				Status:    "error",
			},
		},
	}

	if _, err := engine.IngestTrace(ctx, trace); err != nil {
		t.Fatalf("failed ingesting transient trace: %v", err)
	}

	// Verify node is currently active
	statsInit, _ := engine.GetStats(ctx)
	if statsInit.ActiveNodes == 0 {
		t.Fatalf("expected active nodes, got 0")
	}

	// Simulate passage of 120 hours (5 half-lives: factor = 0.5^5 = 0.03125 < 0.15)
	simulatedFuture := now.Add(120 * time.Hour)

	triggerResp, err := engine.RunConsolidationCycle(ctx, simulatedFuture)
	if err != nil {
		t.Fatalf("consolidation cycle failed: %v", err)
	}

	if triggerResp.NodesArchived == 0 {
		t.Errorf("expected transient nodes to be soft-archived below prune threshold, got %d archived", triggerResp.NodesArchived)
	}

	statsAfter, err := engine.GetStats(ctx)
	if err != nil {
		t.Fatalf("failed retrieving stats: %v", err)
	}
	if statsAfter.ArchivedNodes == 0 {
		t.Errorf("expected archived nodes count > 0 in telemetry stats")
	}

	// Verify archived nodes are excluded from active headers
	headers, err := s.GetAllNodeHeaders(ctx)
	if err != nil {
		t.Fatalf("failed fetching headers: %v", err)
	}
	for _, h := range headers {
		if h.IsArchived {
			t.Errorf("GetAllNodeHeaders returned soft-archived node %s", h.ID)
		}
	}
}

func TestHebbianEdgeWeightReinforcement(t *testing.T) {
	s := createTestStore(t, "test_hebbian.db")
	ctx := context.Background()

	cfg := model.DecayConfig{
		DecayHalfLife:         72 * time.Hour,
		PruneThreshold:        0.10,
		EdgePruneThreshold:    0.05,
		InactivityGracePeriod: 168 * time.Hour,
		HebbianLearningRate:   0.30,
		MaxEdgeWeight:         3.0,
		BatchSize:             100,
	}

	refTime := time.Now().UTC()

	// Initial reinforcement of edge A -> B
	w1, err := s.ReinforceEdge(ctx, "node-A", "node-B", "associates_with", 0.30, cfg.MaxEdgeWeight, refTime)
	if err != nil {
		t.Fatalf("initial ReinforceEdge failed: %v", err)
	}
	if w1 <= 1.0 {
		t.Errorf("expected initial weight > 1.0, got %f", w1)
	}

	// Second reinforcement event
	w2, err := s.ReinforceEdge(ctx, "node-A", "node-B", "associates_with", 0.50, cfg.MaxEdgeWeight, refTime.Add(time.Hour))
	if err != nil {
		t.Fatalf("second ReinforceEdge failed: %v", err)
	}
	if w2 <= w1 {
		t.Errorf("expected weight to increase monotonically (w1=%f, w2=%f)", w1, w2)
	}

	// Reinforce up to upper bound
	for i := 0; i < 10; i++ {
		_, _ = s.ReinforceEdge(ctx, "node-A", "node-B", "associates_with", 1.0, cfg.MaxEdgeWeight, refTime.Add(time.Duration(i)*time.Hour))
	}

	finalWeight, err := s.ReinforceEdge(ctx, "node-A", "node-B", "associates_with", 0.5, cfg.MaxEdgeWeight, refTime.Add(24*time.Hour))
	if err != nil {
		t.Fatalf("final ReinforceEdge failed: %v", err)
	}
	if finalWeight > cfg.MaxEdgeWeight {
		t.Errorf("edge weight exceeded max bound of %f: got %f", cfg.MaxEdgeWeight, finalWeight)
	}
}

func TestComputeImportanceHeuristics(t *testing.T) {
	tests := []struct {
		name        string
		entityType  string
		label       string
		summary     string
		anchors     []string
		minExpected float64
		maxExpected float64
	}{
		{
			name:        "System Config / Architecture Rule",
			entityType:  "concept",
			label:       "Cluster Master Port Config",
			summary:     "INGEST_PORT binding configurations and cluster network bounds",
			anchors:     nil,
			minExpected: 0.85,
			maxExpected: 1.0,
		},
		{
			name:        "Telemetry Rule / Architecture Guideline",
			entityType:  "telemetry_rule",
			label:       "Network Rule",
			summary:     "Cluster architecture rule",
			anchors:     nil,
			minExpected: 0.85,
			maxExpected: 1.0,
		},
		{
			name:        "Task Goal",
			entityType:  "task_goal",
			label:       "Deploy and configure reverse proxy",
			summary:     "Primary deliberation goal",
			anchors:     nil,
			minExpected: 0.85,
			maxExpected: 1.0,
		},
		{
			name:        "Decision",
			entityType:  "decision",
			label:       "Select SQLite for Graph Store",
			summary:     "Persistence engine selection",
			anchors:     nil,
			minExpected: 0.70,
			maxExpected: 0.85,
		},
		{
			name:        "Procedure",
			entityType:  "procedure",
			label:       "Restart Ingest Gateway",
			summary:     "Service lifecycle step",
			anchors:     nil,
			minExpected: 0.70,
			maxExpected: 0.85,
		},
		{
			name:        "Anchor Boost",
			entityType:  "concept",
			label:       "Service Metadata",
			summary:     "General metadata",
			anchors:     []string{"#project:kestrel"},
			minExpected: 0.65, // 0.50 + 0.15
			maxExpected: 1.0,
		},
		{
			name:        "Sensory / Telemetry Fact",
			entityType:  "sensory_fact",
			label:       "Normal CPU thermals",
			summary:     "Telemetry reading",
			anchors:     nil,
			minExpected: 0.40,
			maxExpected: 0.50,
		},
		{
			name:        "Transient Noise / Jitter",
			entityType:  "noise",
			label:       "Transient sensor jitter code_4091",
			summary:     "Ephemeral noise reading",
			anchors:     nil,
			minExpected: 0.20,
			maxExpected: 0.30,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			score := consolidation.ComputeImportance(tc.entityType, tc.label, tc.summary, tc.anchors)
			if score < tc.minExpected || score > tc.maxExpected {
				t.Errorf("ComputeImportance(%s, %s) = %f; expected between %f and %f",
					tc.entityType, tc.label, score, tc.minExpected, tc.maxExpected)
			}
		})
	}
}

func TestExecuteScopedDecay(t *testing.T) {
	s := createTestStore(t, "test_scoped_decay.db")
	ctx := context.Background()
	cfg := model.DefaultDecayConfig()
	engine := consolidation.NewEngine(s, cfg)

	now := time.Now().UTC()

	// Ingest 10 unanchored transient distractor nodes
	distractors := make([]model.Node, 10)
	for i := 0; i < 10; i++ {
		distractors[i] = model.Node{
			ID:              fmt.Sprintf("distractor-%d", i),
			EntityType:      "telemetry",
			Label:           fmt.Sprintf("Transient sensor jitter %d", i),
			Summary:         "Ephemeral telemetry reading",
			ImportanceScore: 0.25,
			StabilityScore:  0.20,
			CreatedAt:       now.Add(-2 * time.Hour),
			LastAccessedAt:  now.Add(-2 * time.Hour),
		}
	}
	if _, err := s.InsertNodes(ctx, distractors); err != nil {
		t.Fatalf("failed inserting distractors: %v", err)
	}

	// Ingest 2 anchored hub nodes
	anchoredHubs := []model.Node{
		{
			ID:              "hub-1",
			EntityType:      "concept",
			Label:           "Cluster Coordinator Hub",
			Summary:         "Core coordinator entity",
			Anchors:         []string{"#project:kestrel"},
			ImportanceScore: 0.40, // Low intrinsic importance, but protected by anchor!
			StabilityScore:  0.30,
			CreatedAt:       now.Add(-2 * time.Hour),
			LastAccessedAt:  now.Add(-2 * time.Hour),
		},
	}
	if _, err := s.InsertNodes(ctx, anchoredHubs); err != nil {
		t.Fatalf("failed inserting anchored hub: %v", err)
	}

	// Ingest 2 high importance config nodes
	criticalConfigs := []model.Node{
		{
			ID:              "config-1",
			EntityType:      "concept",
			Label:           "INGEST_PORT Config Rule",
			Summary:         "Ingest port configuration parameter",
			ImportanceScore: 0.90, // Unanchored, but protected by importance >= 0.80!
			StabilityScore:  0.30,
			CreatedAt:       now.Add(-2 * time.Hour),
			LastAccessedAt:  now.Add(-2 * time.Hour),
		},
	}
	if _, err := s.InsertNodes(ctx, criticalConfigs); err != nil {
		t.Fatalf("failed inserting critical config: %v", err)
	}

	// Trigger accelerated decay: scope="unanchored", grace="0s", decayHalfLife="1h"
	zeroGrace := 0 * time.Second
	oneHour := 1 * time.Hour
	pruneThresh := 0.50

	req := model.DecayRequest{
		Scope:                 "unanchored",
		InactivityGracePeriod: &zeroGrace,
		DecayHalfLife:         &oneHour,
		PruneThreshold:        &pruneThresh,
		MinImportanceToRetain: 0.80,
	}

	resp, err := engine.ExecuteScopedDecay(ctx, req, now)
	if err != nil {
		t.Fatalf("ExecuteScopedDecay failed: %v", err)
	}

	if resp.NodesArchived != 10 {
		t.Errorf("expected 10 distractors archived, got %d", resp.NodesArchived)
	}
	if resp.ProtectedNodes < 2 {
		t.Errorf("expected at least 2 protected nodes, got %d", resp.ProtectedNodes)
	}

	// Verify all distractors are soft-archived
	for i := 0; i < 10; i++ {
		node, err := s.GetNode(ctx, fmt.Sprintf("distractor-%d", i))
		if err != nil || node == nil || !node.IsArchived {
			t.Errorf("distractor-%d was not soft-archived", i)
		}
	}

	// Verify anchored hub and critical config remain active
	hub, err := s.GetNode(ctx, "hub-1")
	if err != nil || hub == nil || hub.IsArchived {
		t.Errorf("hub-1 was improperly archived")
	}

	cfgNode, err := s.GetNode(ctx, "config-1")
	if err != nil || cfgNode == nil || cfgNode.IsArchived {
		t.Errorf("config-1 was improperly archived")
	}
}

func TestExtractorSecretFlag(t *testing.T) {
	extractor := consolidation.NewExtractor(64)

	// 1. Trace with IsSecret: true should mark all derived entities as secret
	traceSecret := model.EpisodicTrace{
		ID:        "trace-sec-01",
		SessionID: "sess-sec-01",
		TaskGoal:  "Generate Master Encryption Key",
		Outcome:   model.OutcomeSuccess,
		IsSecret:  true,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex:   1,
				Thought:     "Exporting master secret key",
				Action:      "export_key",
				Observation: "Key exported successfully",
				Status:      "success",
			},
		},
		SensoryContext: []model.SensoryItem{
			{
				Text:     "Cluster token master-secret-data",
				Salience: 0.8,
			},
		},
	}

	resSecret := extractor.Extract(traceSecret)
	if !resSecret.IsSecret {
		t.Errorf("expected ExtractionResult.IsSecret to be true")
	}
	if len(resSecret.Entities) == 0 {
		t.Fatalf("expected entities to be extracted")
	}
	for _, e := range resSecret.Entities {
		if !e.IsSecret {
			t.Errorf("expected entity %s (%s) to have IsSecret = true", e.ID, e.EntityType)
		}
	}

	// 2. Trace with IsSecret: false but individual step/sensory secret flag
	traceMixed := model.EpisodicTrace{
		ID:        "trace-mixed-01",
		SessionID: "sess-mixed-01",
		TaskGoal:  "Public Deployment Task",
		Outcome:   model.OutcomeSuccess,
		IsSecret:  false,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex:   1,
				Thought:     "Public step with no secrets",
				Action:      "ping_host",
				Status:      "success",
				IsSecret:    false,
			},
			{
				StepIndex:   2,
				Thought:     "Injecting secret password pass123",
				Action:      "inject_cred",
				Status:      "success",
				IsSecret:    true,
			},
		},
		SensoryContext: []model.SensoryItem{
			{
				Text:     "Public telemetry chunk",
				Salience: 0.5,
				IsSecret: false,
			},
			{
				Text:     "Confidential secret credential item",
				Salience: 0.9,
				IsSecret: true,
			},
		},
	}

	resMixed := extractor.Extract(traceMixed)
	if resMixed.IsSecret {
		t.Errorf("expected mixed trace ExtractionResult.IsSecret to be false")
	}
	for _, e := range resMixed.Entities {
		if strings.Contains(e.Summary, "pass123") || strings.Contains(e.Summary, "Confidential secret") {
			if !e.IsSecret {
				t.Errorf("expected secret entity %s to have IsSecret = true", e.ID)
			}
		} else if e.EntityType == "task_goal" {
			if e.IsSecret {
				t.Errorf("expected public task_goal entity to have IsSecret = false")
			}
		}
	}
}

func TestConsolidationSecretPropagationAndRecall(t *testing.T) {
	ctx := context.Background()
	s := createTestStore(t, "test_secret_consolidation.db")

	cipher, err := security.NewAESGCMCipher("")
	if err != nil {
		t.Fatalf("failed to create cipher: %v", err)
	}
	s.SetCipher(cipher)

	recallCfg := recall.DefaultConfig()
	recallCfg.Cipher = cipher
	recallEngine, err := recall.NewEngine(ctx, s, recallCfg)
	if err != nil {
		t.Fatalf("failed to create recall engine: %v", err)
	}

	cfg := model.DefaultDecayConfig()
	consEngine := consolidation.NewEngine(s, cfg)
	consEngine.AddNodeListener(func(nodes []model.Node) {
		recallEngine.RegisterNodes(nodes)
	})

	secretPlaintext := "sk-live-super-secret-token-xyz-12345"
	trace := model.ConsolidateRequest{
		TraceID:     "trace-secret-prop-01",
		SessionID:   "sess-secret-prop",
		TaskGoal:    "Provision Production API Secret Key",
		Outcome:     model.OutcomeSuccess,
		Synchronous: true,
		IsSecret:    true,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex:   1,
				Thought:     "Acquiring credential " + secretPlaintext,
				Action:      "fetch_token",
				Observation: "Token retrieved securely",
				Status:      "success",
			},
		},
		SensoryContext: []model.SensoryItem{
			{
				Text:     "Cluster secret environment seed",
				Salience: 0.9,
			},
		},
	}

	resp, err := consEngine.IngestTrace(ctx, trace)
	if err != nil {
		t.Fatalf("IngestTrace failed: %v", err)
	}
	if resp.Status != "consolidated" {
		t.Fatalf("expected status 'consolidated', got %s", resp.Status)
	}
	if len(resp.CreatedNodes) == 0 {
		t.Fatalf("expected created nodes, got 0")
	}

	// 1. Verify in-memory response nodes carry IsSecret: true
	for _, n := range resp.CreatedNodes {
		if !n.IsSecret {
			t.Errorf("expected created node %s to have IsSecret = true", n.ID)
		}
	}

	// 2. Verify SQLite storage: is_secret = 1 and summary starts with enc:v1:
	activeNodes, err := s.GetAllActiveNodes(ctx)
	if err != nil {
		t.Fatalf("GetAllActiveNodes failed: %v", err)
	}
	if len(activeNodes) == 0 {
		t.Fatalf("expected active nodes in SQLite store, got 0")
	}

	foundSecretNode := false
	for _, n := range activeNodes {
		if !n.IsSecret {
			t.Errorf("expected persistent node %s to have IsSecret = true, got false", n.ID)
		}
		if !strings.HasPrefix(n.Summary, "enc:v1:") {
			t.Errorf("expected node %s summary to be encrypted with 'enc:v1:', got: %s", n.ID, n.Summary)
		}
		if strings.Contains(n.Summary, secretPlaintext) {
			t.Errorf("plaintext secret leaked in SQLite summary: %s", n.Summary)
		}
		if n.EntityType == "decision" {
			foundSecretNode = true
		}
	}
	if !foundSecretNode {
		t.Errorf("expected to find decision node in active nodes")
	}

	// 3. Verify recall without include_secrets masks summary as [REDACTED_SECRET]
	recallRespRedacted, err := recallEngine.Recall(ctx, model.RecallRequest{
		Query:          "Provision Production API Secret Key",
		IncludeSecrets: false,
	})
	if err != nil {
		t.Fatalf("recall failed: %v", err)
	}
	if len(recallRespRedacted.Nodes) == 0 {
		t.Fatalf("expected recall results, got 0")
	}
	for _, sn := range recallRespRedacted.Nodes {
		if !sn.IsSecret {
			t.Errorf("expected recall result %s to have IsSecret = true", sn.ID)
		}
		if sn.Summary != "[REDACTED_SECRET]" {
			t.Errorf("expected redacted summary '[REDACTED_SECRET]', got: %q", sn.Summary)
		}
	}

	// 4. Verify recall with include_secrets: true returns decrypted plaintext summary
	recallRespPlain, err := recallEngine.Recall(ctx, model.RecallRequest{
		Query:          "fetch_token",
		IncludeSecrets: true,
	})
	if err != nil {
		t.Fatalf("recall with include_secrets failed: %v", err)
	}
	if len(recallRespPlain.Nodes) == 0 {
		t.Fatalf("expected recall results with include_secrets, got 0")
	}
	var foundDecrypted bool
	for _, sn := range recallRespPlain.Nodes {
		if !sn.IsSecret {
			t.Errorf("expected recall result %s to have IsSecret = true", sn.ID)
		}
		if strings.HasPrefix(sn.Summary, "enc:v1:") {
			t.Errorf("summary was not decrypted, still has ciphertext prefix: %s", sn.Summary)
		}
		if sn.Summary == "[REDACTED_SECRET]" {
			t.Errorf("summary should not be redacted when include_secrets: true, got: %s", sn.Summary)
		}
		if strings.Contains(sn.Summary, secretPlaintext) {
			foundDecrypted = true
		}
	}
	if !foundDecrypted {
		t.Errorf("expected to find decrypted plaintext %q in results", secretPlaintext)
	}
}

func TestConsolidationSecretBackgroundCycle(t *testing.T) {
	ctx := context.Background()
	s := createTestStore(t, "test_secret_bg_consolidation.db")

	cipher, err := security.NewAESGCMCipher("")
	if err != nil {
		t.Fatalf("failed to create cipher: %v", err)
	}
	s.SetCipher(cipher)

	recallCfg := recall.DefaultConfig()
	recallCfg.Cipher = cipher
	recallEngine, err := recall.NewEngine(ctx, s, recallCfg)
	if err != nil {
		t.Fatalf("failed to create recall engine: %v", err)
	}

	cfg := model.DefaultDecayConfig()
	consEngine := consolidation.NewEngine(s, cfg)
	consEngine.AddNodeListener(func(nodes []model.Node) {
		recallEngine.RegisterNodes(nodes)
	})

	secretData := "vault-bg-token-987654"
	trace := model.ConsolidateRequest{
		TraceID:     "trace-bg-sec-01",
		SessionID:   "sess-bg-sec",
		TaskGoal:    "Background Credential Rotation Pipeline",
		Outcome:     model.OutcomeSuccess,
		Synchronous: false,
		IsSecret:    true,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex:   1,
				Thought:     "Rotating credentials " + secretData,
				Action:      "rotate_credentials",
				Status:      "success",
			},
		},
	}

	// 1. Ingest asynchronously (queues trace)
	resp, err := consEngine.IngestTrace(ctx, trace)
	if err != nil {
		t.Fatalf("IngestTrace failed: %v", err)
	}
	if resp.Status != "accepted" {
		t.Errorf("expected status 'accepted', got %s", resp.Status)
	}

	// 2. Run background consolidation cycle
	cycleResp, err := consEngine.RunConsolidationCycle(ctx, time.Now().UTC())
	if err != nil {
		t.Fatalf("RunConsolidationCycle failed: %v", err)
	}
	if cycleResp.TracesFused != 1 {
		t.Errorf("expected 1 trace fused, got %d", cycleResp.TracesFused)
	}

	// 3. Verify SQLite storage contains encrypted nodes with is_secret = 1
	activeNodes, err := s.GetAllActiveNodes(ctx)
	if err != nil {
		t.Fatalf("GetAllActiveNodes failed: %v", err)
	}
	if len(activeNodes) == 0 {
		t.Fatalf("expected active nodes after consolidation cycle")
	}

	for _, n := range activeNodes {
		if !n.IsSecret {
			t.Errorf("node %s is missing IsSecret = true", n.ID)
		}
		if !strings.HasPrefix(n.Summary, "enc:v1:") {
			t.Errorf("node %s summary is not encrypted: %s", n.ID, n.Summary)
		}
		if strings.Contains(n.Summary, secretData) {
			t.Errorf("plaintext leaked in node %s summary: %s", n.ID, n.Summary)
		}
	}

	// 4. Verify recall masking
	recRedacted, err := recallEngine.Recall(ctx, model.RecallRequest{
		Query:          "Credential Rotation",
		IncludeSecrets: false,
	})
	if err != nil {
		t.Fatalf("recall failed: %v", err)
	}
	if len(recRedacted.Nodes) == 0 {
		t.Fatalf("expected recall results, got 0")
	}
	for _, sn := range recRedacted.Nodes {
		if sn.Summary != "[REDACTED_SECRET]" {
			t.Errorf("expected '[REDACTED_SECRET]', got %q", sn.Summary)
		}
	}

	// 5. Verify recall unmasking
	recPlain, err := recallEngine.Recall(ctx, model.RecallRequest{
		Query:          "Credential Rotation",
		IncludeSecrets: true,
	})
	if err != nil {
		t.Fatalf("recall with include_secrets failed: %v", err)
	}
	var foundPlain bool
	for _, sn := range recPlain.Nodes {
		if strings.Contains(sn.Summary, secretData) {
			foundPlain = true
		}
	}
	if !foundPlain {
		t.Errorf("expected to find plaintext %q in unmasked recall results", secretData)
	}
}
