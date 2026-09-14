package consolidation_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/consolidation"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
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
