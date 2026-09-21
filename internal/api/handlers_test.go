package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/recall"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

type apiMockStore struct {
	nodes   map[string]model.Node
	edges   []model.Edge
	headers []store.NodeHeader
	anchors map[string][]string
}

func newAPIMockStore() *apiMockStore {
	return &apiMockStore{
		nodes:   make(map[string]model.Node),
		anchors: make(map[string][]string),
	}
}

func (m *apiMockStore) Close() error { return nil }

func (m *apiMockStore) AttachAnchors(ctx context.Context, nodeID string, anchors []string) error {
	seen := make(map[string]struct{})
	for _, existing := range m.anchors[nodeID] {
		seen[existing] = struct{}{}
	}
	for _, a := range anchors {
		clean := model.NormalizeAnchor(a)
		if clean != "" {
			if _, exists := seen[clean]; !exists {
				seen[clean] = struct{}{}
				m.anchors[nodeID] = append(m.anchors[nodeID], clean)
			}
		}
	}
	if n, ok := m.nodes[nodeID]; ok {
		n.Anchors = m.anchors[nodeID]
		m.nodes[nodeID] = n
	}
	return nil
}

func (m *apiMockStore) GetAnchorsForNode(ctx context.Context, nodeID string) ([]string, error) {
	return m.anchors[nodeID], nil
}

func (m *apiMockStore) GetNodeIDsForAnchors(ctx context.Context, anchors []string) ([]string, error) {
	var res []string
	seen := make(map[string]struct{})
	target := make(map[string]struct{})
	for _, a := range anchors {
		clean := model.NormalizeAnchor(a)
		if clean != "" {
			target[clean] = struct{}{}
		}
	}
	for nodeID, tags := range m.anchors {
		for _, t := range tags {
			if _, ok := target[t]; ok {
				if _, added := seen[nodeID]; !added {
					seen[nodeID] = struct{}{}
					res = append(res, nodeID)
				}
				break
			}
		}
	}
	return res, nil
}

func (m *apiMockStore) InsertNodes(ctx context.Context, nodes []model.Node) (int, error) {
	for _, n := range nodes {
		if len(n.Anchors) > 0 {
			_ = m.AttachAnchors(ctx, n.ID, n.Anchors)
			n.Anchors = m.anchors[n.ID]
		}
		m.nodes[n.ID] = n
		m.headers = append(m.headers, store.NodeHeader{
			ID:             n.ID,
			EntityType:     n.EntityType,
			Label:          n.Label,
			Summary:        n.Summary,
			Embedding:      n.Embedding,
			LastAccessedAt: n.LastAccessedAt,
			AccessCount:    n.AccessCount,
			StabilityScore: n.StabilityScore,
			Anchors:        m.anchors[n.ID],
		})
	}
	return len(nodes), nil
}

func (m *apiMockStore) InsertEdges(ctx context.Context, edges []model.Edge) (int, error) {
	m.edges = append(m.edges, edges...)
	return len(edges), nil
}

func (m *apiMockStore) GetNode(ctx context.Context, id string) (*model.Node, error) {
	n, exists := m.nodes[id]
	if !exists {
		return nil, nil
	}
	n.Anchors = m.anchors[id]
	return &n, nil
}

func (m *apiMockStore) GetNodes(ctx context.Context, ids []string) (map[string]model.Node, error) {
	res := make(map[string]model.Node)
	for _, id := range ids {
		if n, ok := m.nodes[id]; ok {
			n.Anchors = m.anchors[id]
			res[id] = n
		}
	}
	return res, nil
}

func (m *apiMockStore) GetAllNodeHeaders(ctx context.Context) ([]store.NodeHeader, error) {
	return m.headers, nil
}

func (m *apiMockStore) GetEdgesForNodes(ctx context.Context, nodeIDs []string) ([]model.Edge, error) {
	return m.edges, nil
}

func (m *apiMockStore) RecordAccess(ctx context.Context, nodeIDs []string, accessTime time.Time) error {
	return nil
}

func (m *apiMockStore) GetCounts(ctx context.Context) (int64, int64, error) {
	return int64(len(m.nodes)), int64(len(m.edges)), nil
}

func (m *apiMockStore) GetGraphSummary(ctx context.Context) (*model.GraphSummary, error) {
	return &model.GraphSummary{
		NodeCount:   int64(len(m.nodes)),
		EdgeCount:   int64(len(m.edges)),
		EntityTypes: map[string]int64{"concept": int64(len(m.nodes))},
		AvgDegree:   1.0,
	}, nil
}

func (m *apiMockStore) FindMatchingNode(ctx context.Context, label string, entityType string) (*model.Node, error) {
	for _, n := range m.nodes {
		if n.Label == label {
			return &n, nil
		}
	}
	return nil, nil
}

func (m *apiMockStore) QueueTrace(ctx context.Context, trace model.EpisodicTrace) error {
	return nil
}

func (m *apiMockStore) GetPendingTraces(ctx context.Context, limit int) ([]model.EpisodicTrace, error) {
	return nil, nil
}

func (m *apiMockStore) MarkTraceConsolidated(ctx context.Context, traceID string, consolidatedAt time.Time) error {
	return nil
}

func (m *apiMockStore) ReinforceEdge(ctx context.Context, sourceID, targetID, relationType string, deltaW float64, maxWeight float64, reinforcedAt time.Time) (float64, error) {
	return 1.0 + deltaW, nil
}

func (m *apiMockStore) BoostNodeStability(ctx context.Context, nodeID string, deltaStability float64, reinforcedAt time.Time) error {
	return nil
}

func (m *apiMockStore) ApplyDecayAndPrune(ctx context.Context, cfg model.DecayConfig, refTime time.Time) (decayed int, archived int, prunedEdges int, err error) {
	return 0, 0, 0, nil
}

func (m *apiMockStore) GetConsolidationStats(ctx context.Context) (*model.ConsolidationStats, error) {
	return &model.ConsolidationStats{
		ActiveNodes: int64(len(m.nodes)),
		ActiveEdges: int64(len(m.edges)),
	}, nil
}

func TestAPIServer_Endpoints(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()
	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("engine initialisation failed: %v", err)
	}

	server := NewServer(ms, engine, 8084)

	// 1. Test Insert
	insertPayload := model.InsertRequest{
		Nodes: []model.Node{
			{
				ID:             "fact-1",
				EntityType:     "concept",
				Label:          "Working Memory Buffer",
				Summary:        "Node 2 hosts active scratchpad",
				Embedding:      []float32{0.5, 0.5},
				CreatedAt:      time.Now(),
				LastAccessedAt: time.Now(),
				AccessCount:    1,
				StabilityScore: 1.0,
			},
		},
		Edges: []model.Edge{
			{
				SourceID:     "fact-1",
				TargetID:     "fact-2",
				RelationType: "communicates_with",
				Weight:       1.0,
				CreatedAt:    time.Now(),
			},
		},
	}

	body, _ := json.Marshal(insertPayload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/insert", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var insResp model.InsertResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &insResp)
	if insResp.InsertedNodes != 1 || insResp.InsertedEdges != 1 {
		t.Fatalf("unexpected insert response counts: %+v", insResp)
	}

	// 2. Test Recall
	recallPayload := model.RecallRequest{
		Embedding: []float32{0.5, 0.5},
		TopK:      1,
	}
	body, _ = json.Marshal(recallPayload)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200 on recall, got %d", rec.Code)
	}

	var recResp model.RecallResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &recResp)
	if len(recResp.Nodes) != 1 {
		t.Fatalf("expected 1 recalled node, got %d", len(recResp.Nodes))
	}
	if recResp.Nodes[0].ID != "fact-1" {
		t.Fatalf("expected recalled node fact-1, got %s", recResp.Nodes[0].ID)
	}

	// 3. Test Graph
	req = httptest.NewRequest(http.MethodGet, "/api/v1/memory/graph", nil)
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200 on graph, got %d", rec.Code)
	}

	// 4. Test Health
	req = httptest.NewRequest(http.MethodGet, "/api/v1/memory/health", nil)
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200 on health, got %d", rec.Code)
	}

	var healthResp model.HealthResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &healthResp)
	if healthResp.Status != "healthy" || healthResp.Port != 8084 || healthResp.Node != "sekha-node1" {
		t.Fatalf("unexpected health payload: %+v", healthResp)
	}

	// 5. Test Consolidate Endpoint
	consolidatePayload := model.ConsolidateRequest{
		TraceID:   "test-trace-api",
		SessionID: "sess-api-01",
		TaskGoal:  "Verify consolidation endpoint handler",
		Outcome:   model.OutcomeSuccess,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex: 1,
				Thought:   "Processing step for unit test",
				Action:    "test_action",
				Status:    "success",
			},
		},
	}
	body, _ = json.Marshal(consolidatePayload)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/consolidate", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusAccepted && rec.Code != http.StatusOK {
		t.Fatalf("expected status 202 or 200 on consolidate, got %d", rec.Code)
	}

	// 6. Test Consolidation Stats Endpoint
	req = httptest.NewRequest(http.MethodGet, "/api/v1/memory/consolidation/stats", nil)
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200 on consolidation stats, got %d", rec.Code)
	}
}

func TestAPIServer_ConsolidationRecallEndToEnd(t *testing.T) {
	ms := newAPIMockStore()
	ctx := context.Background()

	// Engine starts with 0 nodes hydrated (empty database at boot)
	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create recall engine: %v", err)
	}

	server := NewServer(ms, engine, 8084)

	// Step 1: Deliberation trace for Project Kestrel is consolidated synchronously
	consolidatePayload := model.ConsolidateRequest{
		TraceID:     "trace-kestrel-01",
		SessionID:   "session-kestrel",
		TaskGoal:    "Project Kestrel ingest service configuration",
		Outcome:     model.OutcomeSuccess,
		Synchronous: true,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex: 1,
				Thought:   "Configure ingest service settings for Project Kestrel",
				Action:    "edit /etc/kestrel/config.yaml",
				Status:    "success",
			},
		},
		SensoryContext: []model.SensoryItem{
			{
				Text:     "Project Kestrel ingest service listening on port 8084",
				Salience: 0.9,
			},
		},
	}

	body, _ := json.Marshal(consolidatePayload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/consolidate", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200 on synchronous consolidate, got %d: %s", rec.Code, rec.Body.String())
	}

	var consResp model.ConsolidateResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &consResp)
	if consResp.NodesFused == 0 && len(consResp.CreatedNodes) == 0 {
		t.Fatalf("expected fused/created nodes, got none: %+v", consResp)
	}

	// Verify in-memory recall index has been populated (Issue B & C resolved)
	if engine.IndexLen() == 0 {
		t.Fatalf("expected in-memory index to contain nodes after consolidation, got 0")
	}

	// Step 2: Query recall with text query ONLY (Issue A & D resolved)
	recallPayload := model.RecallRequest{
		Query: "Project Kestrel ingest service settings",
		TopK:  3,
	}
	body, _ = json.Marshal(recallPayload)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200 on recall, got %d: %s", rec.Code, rec.Body.String())
	}

	var recResp model.RecallResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &recResp)
	if len(recResp.Nodes) == 0 {
		t.Fatalf("expected recall results, got empty list")
	}

	// Top result must have positive SimScore
	topNode := recResp.Nodes[0]
	if topNode.SimScore <= 0.5 {
		t.Fatalf("expected strong semantic SimScore (>0.5) for matching text query, got %f", topNode.SimScore)
	}
	t.Logf("Top recalled node: %s | Label: %s | Sim: %f | Total: %f",
		topNode.ID, topNode.Label, topNode.SimScore, topNode.Score)

	// Step 3: Consolidate distractor trace (Issue E resolved)
	distractorPayload := model.ConsolidateRequest{
		TraceID:     "trace-distractor-01",
		SessionID:   "session-distractor",
		TaskGoal:    "Periodic cooling fan speed adjustment",
		Outcome:     model.OutcomeSuccess,
		Synchronous: true,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex: 1,
				Thought:   "Check thermal sensors and spin fan up to 6000 RPM",
				Action:    "pwm_set 6000",
				Status:    "success",
			},
		},
	}
	body, _ = json.Marshal(distractorPayload)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/consolidate", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	// Query recall again for Kestrel: needle must remain #1 despite recent distractor
	body, _ = json.Marshal(recallPayload)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	var recResp2 model.RecallResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &recResp2)
	if len(recResp2.Nodes) == 0 {
		t.Fatalf("expected recall results, got empty")
	}

	if recResp2.Nodes[0].SimScore <= 0.5 {
		t.Fatalf("expected needle to stay rank 1, but got sim: %f", recResp2.Nodes[0].SimScore)
	}
	t.Logf("Post-distractor Rank 1: %s (sim: %f, score: %f)",
		recResp2.Nodes[0].ID, recResp2.Nodes[0].SimScore, recResp2.Nodes[0].Score)
}

func TestAPIServer_AnchorBoostAndFilter(t *testing.T) {
	ms := newAPIMockStore()
	ctx := context.Background()

	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	server := NewServer(ms, engine, 8084)

	now := time.Now().UTC()

	// 1. Insert two project nodes via /api/v1/memory/insert
	insertReq := model.InsertRequest{
		Nodes: []model.Node{
			{
				ID:             "kestrel-config",
				EntityType:     "decision",
				Label:          "Service Port Config",
				Summary:        "Configured port 8084 for kestrel",
				Anchors:        []string{"#project:kestrel"},
				CreatedAt:      now.Add(-48 * time.Hour),
				LastAccessedAt: now.Add(-48 * time.Hour),
				AccessCount:    1,
				StabilityScore: 1.0,
			},
			{
				ID:             "falcon-config",
				EntityType:     "decision",
				Label:          "Service Port Config",
				Summary:        "Configured port 8084 for falcon",
				Anchors:        []string{"#project:falcon"},
				CreatedAt:      now.Add(-1 * time.Minute),
				LastAccessedAt: now.Add(-1 * time.Minute),
				AccessCount:    50,
				StabilityScore: 1.0,
			},
		},
	}

	body, _ := json.Marshal(insertReq)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/insert", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("insert failed: %d %s", rec.Code, rec.Body.String())
	}

	// 2. Test Recall with anchor_mode: "boost"
	recallBoostPayload := model.RecallRequest{
		Query:      "Service Port Config",
		Anchors:    []string{"#project:kestrel"},
		AnchorMode: "boost",
		TopK:       2,
	}
	body, _ = json.Marshal(recallBoostPayload)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("recall boost failed: %d %s", rec.Code, rec.Body.String())
	}

	var boostResp model.RecallResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &boostResp)
	if len(boostResp.Nodes) != 2 {
		t.Fatalf("expected 2 nodes in boost response, got %d", len(boostResp.Nodes))
	}
	if boostResp.Nodes[0].ID != "kestrel-config" {
		t.Fatalf("expected kestrel-config at rank 1 due to anchor boost, got %s", boostResp.Nodes[0].ID)
	}
	if boostResp.Nodes[0].AnchorScore != 1.0 {
		t.Fatalf("expected anchor_score 1.0 on rank 1 node, got %f", boostResp.Nodes[0].AnchorScore)
	}

	// 3. Test Recall with anchor_mode: "filter"
	recallFilterPayload := model.RecallRequest{
		Query:      "Service Port Config",
		Anchors:    []string{"#project:kestrel"},
		AnchorMode: "filter",
		TopK:       2,
	}
	body, _ = json.Marshal(recallFilterPayload)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("recall filter failed: %d %s", rec.Code, rec.Body.String())
	}

	var filterResp model.RecallResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &filterResp)
	if len(filterResp.Nodes) != 1 {
		t.Fatalf("expected exactly 1 node in filter response, got %d", len(filterResp.Nodes))
	}
	if filterResp.Nodes[0].ID != "kestrel-config" {
		t.Fatalf("expected kestrel-config, got %s", filterResp.Nodes[0].ID)
	}

	// 4. Test Consolidate with anchors
	consolidatePayload := model.ConsolidateRequest{
		TraceID:     "trace-circuit-breaker",
		SessionID:   "sess-cb-01",
		TaskGoal:    "Implement circuit breaker pattern for outbound RPC",
		Outcome:     model.OutcomeSuccess,
		Anchors:     []string{"#pattern:circuit-breaker"},
		Synchronous: true,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex: 1,
				Thought:   "Wrap RPC calls in resilient circuit breaker",
				Action:    "init_circuit_breaker",
				Status:    "success",
			},
		},
	}
	body, _ = json.Marshal(consolidatePayload)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/consolidate", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("consolidate failed: %d %s", rec.Code, rec.Body.String())
	}

	// 5. Test Recall for newly consolidated node with anchor filter
	recallCbPayload := model.RecallRequest{
		Query:      "circuit breaker resilient",
		Anchors:    []string{"#pattern:circuit-breaker"},
		AnchorMode: "filter",
		TopK:       5,
	}
	body, _ = json.Marshal(recallCbPayload)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("recall cb failed: %d %s", rec.Code, rec.Body.String())
	}

	var cbResp model.RecallResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &cbResp)
	if len(cbResp.Nodes) == 0 {
		t.Fatalf("expected recalled nodes for circuit breaker anchor, got 0")
	}
	for _, n := range cbResp.Nodes {
		var found bool
		for _, a := range n.Anchors {
			if a == "#pattern:circuit-breaker" {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected node %s to have #pattern:circuit-breaker, got %v", n.ID, n.Anchors)
		}
	}
}
