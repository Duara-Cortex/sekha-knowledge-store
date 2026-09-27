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

func TestDynamicEntityScaling_TraceLengthsAndTopics(t *testing.T) {
	s := createTestStore(t, "test_scaling.db")
	cfg := model.DefaultDecayConfig()
	engine := consolidation.NewEngine(s, cfg)
	ctx := context.Background()

	// 1. Networking Logs (~1 KB trace)
	var netLogs strings.Builder
	netLogs.WriteString("10:00:01.000 [NET] interface eth0 link up 1000Mbps full duplex\n")
	netLogs.WriteString("10:00:01.100 [TCP] inbound connection from 192.168.1.150:54321 to 192.168.1.10:8084\n")
	netLogs.WriteString("10:00:01.150 [TLS] handshake started cipher TLS_AES_256_GCM_SHA384 curve X25519\n")
	netLogs.WriteString("10:00:01.200 [TLS] handshake completed session_resumed=false\n")
	netLogs.WriteString("10:00:01.300 [HTTP2] stream 1 opened headers :method=POST :path=/api/v1/memory/recall\n")
	netLogs.WriteString("10:00:01.400 [ROUTER] route matched cluster_ingress rule priority=100\n")
	netLogs.WriteString("10:00:01.500 [NET] packet drop detected queue_id=2 rx_dropped=3 buffer_overflow\n")
	netLogs.WriteString("10:00:01.600 [TCP] connection closed fin_wait_2 duration=500ms bytes_sent=1284\n")
	for netLogs.Len() < 1024 {
		netLogs.WriteString("10:00:02.000 [NET] keepalive probe sent to peer router_bgp_neighbor=192.168.1.1\n")
	}

	trace1KB := model.ConsolidateRequest{
		TraceID:        "trace-net-1kb",
		SessionID:      "sess-net-01",
		TaskGoal:       "Analyze edge cluster networking logs and diagnose packet drops",
		Outcome:        model.OutcomeSuccess,
		ExecutionTrace: netLogs.String(),
		Synchronous:    true,
	}

	resp1KB, err := engine.IngestTrace(ctx, trace1KB)
	if err != nil {
		t.Fatalf("failed ingesting 1 KB trace: %v", err)
	}

	t.Logf("1 KB Networking trace: %d entities extracted, %d nodes fused",
		resp1KB.EntitiesExtracted, resp1KB.NodesFused)

	// 2. Parquet Serialization (~10 KB trace)
	var parquetLogs strings.Builder
	parquetLogs.WriteString("09:00:00 [PARQUET_WRITER] Initialising file schema version=2.6 num_columns=12\n")
	parquetLogs.WriteString("09:00:00 [SCHEMA] column 0: timestamp type=INT64 logical_type=TIMESTAMP_MILLIS\n")
	parquetLogs.WriteString("09:00:00 [SCHEMA] column 1: sensor_id type=BYTE_ARRAY logical_type=UTF8 dictionary=true\n")
	parquetLogs.WriteString("09:00:00 [SCHEMA] column 2: cpu_temp type=FLOAT encoding=PLAIN compression=SNAPPY\n")
	parquetLogs.WriteString("09:00:00 [SCHEMA] column 3: memory_used type=INT64 encoding=RLE_DICTIONARY\n")
	for parquetLogs.Len() < 10*1024 {
		idx := parquetLogs.Len() / 120
		parquetLogs.WriteString(fmt.Sprintf("09:00:%02d [ROW_GROUP_%d] written 25000 records compressed_size=84200 uncompressed_size=164000 snappy_ratio=0.51 crc32=98af12\n",
			idx%60, idx))
		parquetLogs.WriteString(fmt.Sprintf("09:00:%02d [DICTIONARY_%d] column='sensor_id' unique_entries=450 page_offset=%d\n",
			idx%60, idx, idx*4096))
	}

	trace10KB := model.ConsolidateRequest{
		TraceID:        "trace-parquet-10kb",
		SessionID:      "sess-parquet-01",
		TaskGoal:       "Validate Parquet columnar serialization and Snappy dictionary compression",
		Outcome:        model.OutcomeSuccess,
		ExecutionTrace: parquetLogs.String(),
		Synchronous:    true,
	}

	resp10KB, err := engine.IngestTrace(ctx, trace10KB)
	if err != nil {
		t.Fatalf("failed ingesting 10 KB trace: %v", err)
	}

	t.Logf("10 KB Parquet trace: %d entities extracted, %d nodes fused",
		resp10KB.EntitiesExtracted, resp10KB.NodesFused)

	// 3. Cooling Telemetry (~90 KB trace)
	var coolingLogs strings.Builder
	coolingLogs.WriteString("12:00:00.000 [THERMAL_CONTROLLER] Initialised active PID cooling control loop\n")
	coolingLogs.WriteString("12:00:00.010 [CONFIG] target_temperature=65.0C max_safe_temp=85.0C min_pwm=2000rpm\n")
	coolingLogs.WriteString("12:00:00.050 [SENSOR_CALIBRATION] zone0=cpu_core zone1=gpu_embedded zone2=ambient_exhaust\n")
	for coolingLogs.Len() < 90*1024 {
		sec := (coolingLogs.Len() / 200) % 3600
		temp := 60.0 + float64(sec%25)
		pwm := 2500 + (sec % 3500)
		coolingLogs.WriteString(fmt.Sprintf("12:%02d:%02d [SAMPLE_%d] zone0_temp=%.2fC zone1_temp=%.2fC fan_pwm=%drpm pid_err=%.2f duty_cycle=0.%d throttling=nominal\n",
			sec/60, sec%60, sec, temp, temp-5.0, pwm, temp-65.0, (pwm*100)/6000))
		coolingLogs.WriteString(fmt.Sprintf("12:%02d:%02d [ALERT_POLICY] check_sensor_deviation delta=%.2fC hysteresis=0.5C cooling_channel=%d\n",
			sec/60, sec%60, temp-55.0, sec%4))
	}
	coolingLogs.WriteString("12:59:59.900 [FAN_GOVERNOR] ERROR fan1 tachometer failed, forcing max duty on cooling_channel=1\n")

	trace90KB := model.ConsolidateRequest{
		TraceID:        "trace-cooling-90kb",
		SessionID:      "sess-cooling-01",
		TaskGoal:       "Monitor edge cluster cooling telemetry and adaptive PWM fan governor",
		Outcome:        model.OutcomeSuccess,
		ExecutionTrace: coolingLogs.String(),
		Synchronous:    true,
	}

	resp90KB, err := engine.IngestTrace(ctx, trace90KB)
	if err != nil {
		t.Fatalf("failed ingesting 90 KB trace: %v", err)
	}

	t.Logf("90 KB Cooling trace: %d entities extracted, %d nodes fused",
		resp90KB.EntitiesExtracted, resp90KB.NodesFused)

	// Confirm entity counts vary with trace content. Counts are not required to be monotonic
	// across topics: a long but repetitive trace can legitimately yield fewer concepts.
	counts := map[int]bool{resp1KB.EntitiesExtracted: true, resp10KB.EntitiesExtracted: true, resp90KB.EntitiesExtracted: true}
	if len(counts) != 3 {
		t.Errorf("entity counts did not vary across traces: 1KB=%d, 10KB=%d, 90KB=%d",
			resp1KB.EntitiesExtracted, resp10KB.EntitiesExtracted, resp90KB.EntitiesExtracted)
	}
	for name, n := range map[string]int{"1KB": resp1KB.EntitiesExtracted, "10KB": resp10KB.EntitiesExtracted, "90KB": resp90KB.EntitiesExtracted} {
		if n == 11 {
			t.Errorf("fixed 11-entity count detected for %s trace", name)
		}
	}

	// Confirm extracted concepts reflect each trace's subject matter.
	assertConcepts(t, "networking", resp1KB.CreatedNodes, "tls", "packet", "handshake")
	assertConcepts(t, "parquet", resp10KB.CreatedNodes, "parquet_writer", "snappy", "dictionary")
	assertConcepts(t, "cooling", resp90KB.CreatedNodes, "fan_pwm", "pid_err", "thermal_controller")

	// Confirm the single injected failure line is captured as an incident despite sampling.
	foundIncident := false
	for _, n := range resp90KB.CreatedNodes {
		if n.EntityType == "incident" && strings.Contains(n.Label, "tachometer failed") {
			foundIncident = true
		}
	}
	if !foundIncident {
		t.Errorf("expected ERROR line in 90 KB cooling trace to be extracted as an incident")
	}
}

// assertConcepts checks that the wanted concepts were extracted and that no concept is a
// value-like token (timestamps, readings, digests) rather than a domain term.
func assertConcepts(t *testing.T, topic string, nodes []model.Node, want ...string) {
	t.Helper()
	concepts := make(map[string]bool)
	for _, n := range nodes {
		if n.EntityType != "concept" {
			continue
		}
		concepts[n.Label] = true
		letters := 0
		for _, c := range n.Label {
			if c >= 'a' && c <= 'z' {
				letters++
			}
		}
		if letters < 3 || strings.HasSuffix(n.Label, "rpm") || strings.Contains(n.Label, ":") {
			t.Errorf("%s trace: value-like token extracted as concept: %q", topic, n.Label)
		}
	}
	for _, w := range want {
		if !concepts[w] {
			t.Errorf("%s trace: expected concept %q to be extracted", topic, w)
		}
	}
}

func TestGraphFusion_OverlappingEntitiesDuplicatePrevention(t *testing.T) {
	s := createTestStore(t, "test_dedup_overlap.db")
	cfg := model.DefaultDecayConfig()
	engine := consolidation.NewEngine(s, cfg)
	ctx := context.Background()

	// Ingest trace with repeatedly referenced concept "snappy_compression"
	trace := model.ConsolidateRequest{
		TraceID:     "trace-dedup-01",
		SessionID:   "sess-dedup",
		TaskGoal:    "Verify Snappy compression in storage engine",
		Outcome:     model.OutcomeSuccess,
		Synchronous: true,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex:   1,
				Thought:     "Testing snappy_compression performance",
				Action:      "apply_snappy_compression",
				Observation: "snappy_compression succeeded",
				Status:      "success",
			},
			{
				StepIndex:   2,
				Thought:     "Benchmarking snappy_compression throughput",
				Action:      "verify_snappy_compression",
				Observation: "snappy_compression ratio verified",
				Status:      "success",
			},
		},
		SensoryContext: []model.SensoryItem{
			{
				Text:     "Active telemetry verifies snappy_compression is operational",
				Salience: 0.9,
			},
		},
	}

	resp, err := engine.IngestTrace(ctx, trace)
	if err != nil {
		t.Fatalf("failed ingesting trace: %v", err)
	}

	t.Logf("Deduplication test: %d entities extracted, %d nodes fused",
		resp.EntitiesExtracted, resp.NodesFused)

	// Verify only ONE node with label "snappy_compression" exists in SQLite
	nodes, err := s.GetAllActiveNodes(ctx)
	if err != nil {
		t.Fatalf("failed querying active nodes: %v", err)
	}
	snappyCount := 0
	for _, n := range nodes {
		if strings.ToLower(n.Label) == "snappy_compression" {
			snappyCount++
		}
	}

	if snappyCount != 1 {
		t.Errorf("expected exactly 1 canonical node for 'snappy_compression', got %d", snappyCount)
	}
}


func TestGraphFusion_CrossTraceDuplicatePrevention(t *testing.T) {
	s := createTestStore(t, "test_dedup_cross_trace.db")
	engine := consolidation.NewEngine(s, model.DefaultDecayConfig())
	ctx := context.Background()

	logs := "10:00:01 [PARQUET_WRITER] snappy compression enabled for dictionary pages\n" +
		"10:00:02 [PARQUET_WRITER] row group flushed with snappy dictionary encoding\n"
	first := model.ConsolidateRequest{
		TraceID: "trace-cross-01", SessionID: "sess-cross-01",
		TaskGoal: "Validate Parquet snappy compression", Outcome: model.OutcomeSuccess,
		ExecutionTrace: logs, Synchronous: true,
	}
	second := first
	second.TraceID, second.SessionID = "trace-cross-02", "sess-cross-02"

	if _, err := engine.IngestTrace(ctx, first); err != nil {
		t.Fatalf("failed ingesting first trace: %v", err)
	}
	before, err := s.GetAllActiveNodes(ctx)
	if err != nil {
		t.Fatalf("failed querying nodes: %v", err)
	}
	resp, err := engine.IngestTrace(ctx, second)
	if err != nil {
		t.Fatalf("failed ingesting second trace: %v", err)
	}
	after, err := s.GetAllActiveNodes(ctx)
	if err != nil {
		t.Fatalf("failed querying nodes: %v", err)
	}

	if len(after) != len(before) {
		t.Errorf("expected overlapping trace to fuse into existing nodes: %d nodes before, %d after", len(before), len(after))
	}
	if len(resp.CreatedNodes) != 0 || resp.NodesFused != resp.EntitiesExtracted {
		t.Errorf("expected all %d entities to fuse into existing nodes, got %d created, %d fused",
			resp.EntitiesExtracted, len(resp.CreatedNodes), resp.NodesFused)
	}

	labels := make(map[string]int)
	for _, n := range after {
		labels[n.EntityType+"::"+strings.ToLower(n.Label)]++
	}
	for key, n := range labels {
		if n > 1 {
			t.Errorf("duplicate node %q stored %d times", key, n)
		}
	}
}

func TestFusion_InBatchDuplicateEntitiesShareCanonicalNode(t *testing.T) {
	s := createTestStore(t, "test_dedup_in_batch.db")
	fusion := consolidation.NewFusionEngine(s, model.DefaultDecayConfig())
	ctx := context.Background()

	extraction := consolidation.ExtractionResult{
		Salience: 0.8,
		Entities: []model.ExtractedEntity{
			{ID: "goal-a", EntityType: "task_goal", Label: "Tune cooling loop", Salience: 0.8, ImportanceScore: 0.85},
			{ID: "evt-a", EntityType: "incident", Label: "[FAN] tachometer failed", Salience: 0.6, ImportanceScore: 0.7},
			{ID: "evt-b", EntityType: "incident", Label: "[fan] Tachometer failed ", Salience: 0.6, ImportanceScore: 0.8},
		},
		Edges: []model.ExtractedRelation{
			{SourceID: "evt-a", TargetID: "goal-a", RelationType: "context_for", Weight: 0.7},
			{SourceID: "evt-b", TargetID: "goal-a", RelationType: "context_for", Weight: 0.7},
		},
	}

	res, err := fusion.Fuse(ctx, extraction, time.Now().UTC())
	if err != nil {
		t.Fatalf("fusion failed: %v", err)
	}
	if res.EntitiesCreated != 2 || res.EntitiesFused != 1 {
		t.Errorf("expected 2 created and 1 fused, got %d created and %d fused", res.EntitiesCreated, res.EntitiesFused)
	}
	if res.EntityIDMappings["evt-b"] != "evt-a" {
		t.Errorf("expected evt-b to map to canonical evt-a, got %q", res.EntityIDMappings["evt-b"])
	}

	nodes, err := s.GetAllActiveNodes(ctx)
	if err != nil {
		t.Fatalf("failed querying nodes: %v", err)
	}
	if len(nodes) != 2 {
		t.Fatalf("expected 2 stored nodes, got %d", len(nodes))
	}
	for _, n := range nodes {
		if n.ID == "evt-a" && n.ImportanceScore != 0.8 {
			t.Errorf("expected canonical node to take the higher importance 0.8, got %v", n.ImportanceScore)
		}
	}
}

func TestExecutionTrace_RepeatedIncidentsDoNotExhaustEventBudget(t *testing.T) {
	s := createTestStore(t, "test_retry_incidents.db")
	engine := consolidation.NewEngine(s, model.DefaultDecayConfig())
	ctx := context.Background()

	var logs strings.Builder
	for i := 0; i < 100; i++ {
		logs.WriteString(fmt.Sprintf("10:00:%02d.%03d [UPLINK] ERROR connection refused by 10.0.0.7:9000, retrying\n", i%60, i))
		logs.WriteString(fmt.Sprintf("10:00:%02d.%03d [QUEUE_%d] backlog depth=%d consumer_lag=%dms\n", i%60, i, i, 100+i, 20*i))
	}

	resp, err := engine.IngestTrace(ctx, model.ConsolidateRequest{
		TraceID: "trace-retry-loop", SessionID: "sess-retry",
		TaskGoal: "Diagnose uplink retry storm", Outcome: model.OutcomeFailure,
		ExecutionTrace: logs.String(), Synchronous: true,
	})
	if err != nil {
		t.Fatalf("failed ingesting trace: %v", err)
	}

	incidents, events := 0, 0
	for _, n := range resp.CreatedNodes {
		switch n.EntityType {
		case "incident":
			incidents++
		case "telemetry", "procedure", "episodic_event":
			events++
		}
	}
	if incidents != 1 {
		t.Errorf("expected repeated ERROR lines to collapse into 1 incident, got %d", incidents)
	}
	if events == 0 {
		t.Errorf("expected non-incident lines to be sampled alongside the repeated incident")
	}
}

// TestNode2ShapedPayload_EntityCountsVary covers the trajectory + sensory payload shape sent by
// Node 2 (no execution_trace), which is the shape that previously always produced 11 entities.
func TestNode2ShapedPayload_EntityCountsVary(t *testing.T) {
	s := createTestStore(t, "test_node2_shape.db")
	engine := consolidation.NewEngine(s, model.DefaultDecayConfig())
	ctx := context.Background()

	longObservation := strings.Repeat("Parquet writer flushed row group with snappy dictionary pages; column statistics min max null_count recorded; footer checksum verified. ", 20)
	longSensory := strings.Repeat("Edge cooling telemetry: fan governor raised pwm duty as cpu die temperature crossed hysteresis band while ambient intake remained stable. ", 20)

	cases := []struct {
		name        string
		goal        string
		thought     string
		observation string
		sensory     string
	}{
		{"networking", "Diagnose packet loss on edge uplink", "Inspect interface counters for rx drops",
			"rx_dropped counter rising on eth0 queue 2 due to ring buffer overflow under burst traffic from ingress router", "BGP neighbor session flapped twice in the last hour"},
		{"parquet-long-observation", "Validate Parquet serialisation", "Check writer output", longObservation, "Writer completed"},
		{"cooling-long-sensory", "Tune cooling governor", "Review thermal telemetry", "Governor adjusted duty cycle", longSensory},
	}

	counts := make(map[string]int)
	for i, c := range cases {
		resp, err := engine.IngestTrace(ctx, model.ConsolidateRequest{
			TraceID: fmt.Sprintf("trace-node2-%d", i), SessionID: fmt.Sprintf("sess-node2-%d", i),
			TaskGoal: c.goal, Outcome: model.OutcomeSuccess, Synchronous: true,
			Trajectory: []model.TrajectoryStep{{
				StepIndex: 1, Thought: c.thought, Action: "analyse", Observation: c.observation, Status: "success",
			}},
			SensoryContext: []model.SensoryItem{{Text: c.sensory, Salience: 0.8}},
		})
		if err != nil {
			t.Fatalf("%s: failed ingesting trace: %v", c.name, err)
		}
		t.Logf("%s: %d entities extracted", c.name, resp.EntitiesExtracted)
		counts[c.name] = resp.EntitiesExtracted
	}

	distinct := make(map[int]bool)
	for _, n := range counts {
		distinct[n] = true
	}
	if len(distinct) != len(cases) {
		t.Errorf("expected entity counts to vary across Node 2 payloads, got %v", counts)
	}
	for _, name := range []string{"parquet-long-observation", "cooling-long-sensory"} {
		if counts[name] <= 11 {
			t.Errorf("%s: expected long payload to exceed the old 11-entity count, got %d", name, counts[name])
		}
	}
}
