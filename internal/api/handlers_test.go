package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/embedding"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/recall"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/security"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

type apiMockStore struct {
	nodes              map[string]model.Node
	edges              []model.Edge
	headers            []store.NodeHeader
	anchors            map[string][]string
	applyScopedDecayFn func(ctx context.Context, req model.DecayRequest, refTime time.Time) (int, int, int, int, error)
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
			ID:              n.ID,
			EntityType:      n.EntityType,
			Label:           n.Label,
			Summary:         n.Summary,
			Embedding:       n.Embedding,
			LastAccessedAt:  n.LastAccessedAt,
			AccessCount:     n.AccessCount,
			StabilityScore:  n.StabilityScore,
			ImportanceScore: n.ImportanceScore,
			IsSecret:        n.IsSecret,
			Anchors:         m.anchors[n.ID],
		})
	}
	return len(nodes), nil
}

func (m *apiMockStore) UpsertNode(ctx context.Context, node model.Node) error {
	_, err := m.InsertNodes(ctx, []model.Node{node})
	return err
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

func (m *apiMockStore) GetAllActiveNodes(ctx context.Context) ([]model.Node, error) {
	res := make([]model.Node, 0, len(m.nodes))
	for _, n := range m.nodes {
		if !n.IsArchived {
			n.Anchors = m.anchors[n.ID]
			res = append(res, n)
		}
	}
	return res, nil
}

func (m *apiMockStore) GetEdgesForNodes(ctx context.Context, nodeIDs []string) ([]model.Edge, error) {
	if len(nodeIDs) == 0 {
		return nil, nil
	}
	idMap := make(map[string]bool, len(nodeIDs))
	for _, id := range nodeIDs {
		idMap[id] = true
	}
	var res []model.Edge
	for _, e := range m.edges {
		if idMap[e.SourceID] || idMap[e.TargetID] {
			res = append(res, e)
		}
	}
	return res, nil
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

func (m *apiMockStore) ApplyScopedDecay(ctx context.Context, req model.DecayRequest, refTime time.Time) (decayed int, archived int, prunedEdges int, protected int, err error) {
	if m.applyScopedDecayFn != nil {
		return m.applyScopedDecayFn(ctx, req, refTime)
	}
	return 0, 0, 0, 0, nil
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

func generate64DimVector(seed int) []float32 {
	vec := make([]float32, 64)
	var sum float64
	for i := 0; i < 64; i++ {
		// Generate varied fractional float32 values
		v := float32(math.Sin(float64(seed*67 + i*31)))
		vec[i] = v
		sum += float64(v * v)
	}
	mag := float32(math.Sqrt(sum))
	if mag > 0 {
		for i := 0; i < 64; i++ {
			vec[i] /= mag
		}
	}
	return vec
}

func TestRecall_EmbeddingOmissionByDefaultAndFlag(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()
	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("engine initialisation failed: %v", err)
	}
	server := NewServer(ms, engine, 8084)

	vec64 := generate64DimVector(42)
	node := model.Node{
		ID:             "node-emb-test",
		EntityType:     "concept",
		Label:          "Embedding Test Node",
		Summary:        "Node with 64-dimensional float vector",
		Embedding:      vec64,
		CreatedAt:      time.Now(),
		LastAccessedAt: time.Now(),
		AccessCount:    10,
		StabilityScore: 1.0,
	}
	_, _ = ms.InsertNodes(ctx, []model.Node{node})
	engine.RegisterNodes([]model.Node{node})

	// 1. Default (include_embeddings=false): raw float vector MUST be omitted
	recallReq := model.RecallRequest{
		Query: "Embedding Test Node",
		TopK:  1,
	}
	body, _ := json.Marshal(recallReq)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("recall failed with status %d: %s", rec.Code, rec.Body.String())
	}

	// Verify in typed RecallResponse that Embedding is nil / empty
	var respDefault model.RecallResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &respDefault)
	if len(respDefault.Nodes) == 0 {
		t.Fatalf("expected recalled node, got 0")
	}
	if len(respDefault.Nodes[0].Embedding) != 0 {
		t.Fatalf("expected nil/empty embedding by default, got %d elements", len(respDefault.Nodes[0].Embedding))
	}

	// Verify in raw JSON map that "embedding" key is completely omitted
	var rawMapDefault map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &rawMapDefault)
	nodesList, ok := rawMapDefault["nodes"].([]any)
	if !ok || len(nodesList) == 0 {
		t.Fatalf("failed parsing nodes in raw response: %+v", rawMapDefault)
	}
	nodeObj := nodesList[0].(map[string]any)
	if _, hasEmb := nodeObj["embedding"]; hasEmb {
		t.Fatalf("expected 'embedding' key to be omitted from JSON by default, but it was present: %+v", nodeObj)
	}

	// 2. Explicit include_embeddings=true in JSON body: raw 64-D float vector MUST be returned
	recallReqTrue := model.RecallRequest{
		Query:             "Embedding Test Node",
		TopK:              1,
		IncludeEmbeddings: true,
	}
	body, _ = json.Marshal(recallReqTrue)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("recall with include_embeddings=true failed with status %d: %s", rec.Code, rec.Body.String())
	}

	var respTrue model.RecallResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &respTrue)
	if len(respTrue.Nodes) == 0 {
		t.Fatalf("expected recalled node, got 0")
	}
	if len(respTrue.Nodes[0].Embedding) != 64 {
		t.Fatalf("expected complete 64-D vector when include_embeddings=true, got %d elements", len(respTrue.Nodes[0].Embedding))
	}

	var rawMapTrue map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &rawMapTrue)
	nodeObjTrue := rawMapTrue["nodes"].([]any)[0].(map[string]any)
	embRaw, hasEmb := nodeObjTrue["embedding"]
	if !hasEmb {
		t.Fatalf("expected 'embedding' key to be present in JSON when include_embeddings=true")
	}
	embArr, isArr := embRaw.([]any)
	if !isArr || len(embArr) != 64 {
		t.Fatalf("expected 64 elements in raw JSON embedding, got %v", embRaw)
	}

	// 3. URL query parameter ?include_embeddings=true
	body, _ = json.Marshal(recallReq)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall?include_embeddings=true", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("recall with query param failed: %d: %s", rec.Code, rec.Body.String())
	}

	var respQueryParam model.RecallResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &respQueryParam)
	if len(respQueryParam.Nodes[0].Embedding) != 64 {
		t.Fatalf("expected 64-D vector via query param, got %d", len(respQueryParam.Nodes[0].Embedding))
	}
}

func TestRecall_FieldProjection(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()
	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("engine initialisation failed: %v", err)
	}
	server := NewServer(ms, engine, 8084)

	vec64 := generate64DimVector(101)
	node := model.Node{
		ID:             "node-projection-test",
		EntityType:     "decision",
		Label:          "Buffer Allocation Strategy",
		Summary:        "Fixed ring-buffer with sliding window eviction",
		Embedding:      vec64,
		Anchors:        []string{"#layer:memory"},
		CreatedAt:      time.Now(),
		LastAccessedAt: time.Now(),
		AccessCount:    25,
		StabilityScore: 0.95,
	}
	_, _ = ms.InsertNodes(ctx, []model.Node{node})
	engine.RegisterNodes([]model.Node{node})

	// 1. Projection with fields=["id", "label", "score"] in JSON body
	recallReq := model.RecallRequest{
		Query:  "Buffer Allocation Strategy",
		TopK:   1,
		Fields: []string{"id", "label", "score"},
	}
	body, _ := json.Marshal(recallReq)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}

	var rawMap map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &rawMap)
	nodeObj := rawMap["nodes"].([]any)[0].(map[string]any)

	// Assert that ONLY projected fields exist (id, label, score)
	expectedKeys := map[string]bool{"id": true, "label": true, "score": true}
	for k := range nodeObj {
		if !expectedKeys[k] {
			t.Fatalf("unexpected projected field '%s' present in node: %+v", k, nodeObj)
		}
	}
	if len(nodeObj) != 3 {
		t.Fatalf("expected exactly 3 fields, got %d: %+v", len(nodeObj), nodeObj)
	}
	if nodeObj["id"] != "node-projection-test" {
		t.Fatalf("expected id 'node-projection-test', got %v", nodeObj["id"])
	}
	if nodeObj["label"] != "Buffer Allocation Strategy" {
		t.Fatalf("expected label 'Buffer Allocation Strategy', got %v", nodeObj["label"])
	}
	if _, hasScore := nodeObj["score"]; !hasScore {
		t.Fatalf("expected 'score' field in projected node")
	}

	// 2. Projection with fields as comma-separated string in query parameter: ?fields=id,label,score
	recallReqBase := model.RecallRequest{
		Query: "Buffer Allocation Strategy",
		TopK:  1,
	}
	body, _ = json.Marshal(recallReqBase)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall?fields=id,label,score", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}

	var rawMapQuery map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &rawMapQuery)
	nodeObjQuery := rawMapQuery["nodes"].([]any)[0].(map[string]any)
	if len(nodeObjQuery) != 3 {
		t.Fatalf("expected exactly 3 fields from query param projection, got %d: %+v", len(nodeObjQuery), nodeObjQuery)
	}
	for k := range nodeObjQuery {
		if !expectedKeys[k] {
			t.Fatalf("unexpected field '%s' in query param projection: %+v", k, nodeObjQuery)
		}
	}

	// 3. Projection with fields as comma-separated string in JSON body: {"fields": "id,label,score"}
	rawJSONBody := []byte(`{"query": "Buffer Allocation Strategy", "top_k": 1, "fields": "id,label,score"}`)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(rawJSONBody))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}

	var rawMapStringField map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &rawMapStringField)
	nodeObjStringField := rawMapStringField["nodes"].([]any)[0].(map[string]any)
	if len(nodeObjStringField) != 3 {
		t.Fatalf("expected exactly 3 fields from string field projection, got %d: %+v", len(nodeObjStringField), nodeObjStringField)
	}

	// 4. Projection with 5 fields: ["id", "label", "summary", "anchors", "score"]
	recallReq5 := model.RecallRequest{
		Query:  "Buffer Allocation Strategy",
		TopK:   1,
		Fields: []string{"id", "label", "summary", "anchors", "score"},
	}
	body, _ = json.Marshal(recallReq5)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}

	var rawMap5 map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &rawMap5)
	nodeObj5 := rawMap5["nodes"].([]any)[0].(map[string]any)
	if len(nodeObj5) != 5 {
		t.Fatalf("expected 5 fields, got %d: %+v", len(nodeObj5), nodeObj5)
	}
	expected5 := map[string]bool{"id": true, "label": true, "summary": true, "anchors": true, "score": true}
	for k := range nodeObj5 {
		if !expected5[k] {
			t.Fatalf("unexpected field '%s' in 5-field projection: %+v", k, nodeObj5)
		}
	}

	// 5. Projection explicitly requesting embedding: ["id", "embedding"]
	recallReqEmb := model.RecallRequest{
		Query:  "Buffer Allocation Strategy",
		TopK:   1,
		Fields: []string{"id", "embedding"},
	}
	body, _ = json.Marshal(recallReqEmb)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}

	var rawMapEmb map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &rawMapEmb)
	nodeObjEmb := rawMapEmb["nodes"].([]any)[0].(map[string]any)
	if len(nodeObjEmb) != 2 {
		t.Fatalf("expected 2 fields (id, embedding), got %d: %+v", len(nodeObjEmb), nodeObjEmb)
	}
	if _, hasEmb := nodeObjEmb["embedding"]; !hasEmb {
		t.Fatalf("expected 'embedding' field when explicitly requested in fields projection")
	}
}

func TestRecall_PayloadSizeBenchmark(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()
	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("engine initialisation failed: %v", err)
	}
	server := NewServer(ms, engine, 8084)

	now := time.Now().UTC()
	var testNodes []model.Node
	for i := 1; i <= 30; i++ {
		vec := generate64DimVector(i * 13)
		testNodes = append(testNodes, model.Node{
			ID:             fmt.Sprintf("node-bench-%03d", i),
			EntityType:     "concept",
			Label:          fmt.Sprintf("Entity #%d Buffer", i),
			Summary:        fmt.Sprintf("State replication config for node #%d.", i),
			Anchors:        []string{"#kestrel"},
			Embedding:      vec,
			CreatedAt:      now.Add(-time.Duration(i) * time.Hour),
			LastAccessedAt: now.Add(-time.Duration(i) * time.Minute),
			AccessCount:    int64(10 + i*2),
			StabilityScore: 0.9,
		})
	}
	_, _ = ms.InsertNodes(ctx, testNodes)
	engine.RegisterNodes(testNodes)

	// Ingest edges connecting consecutive nodes
	var testEdges []model.Edge
	for i := 1; i < len(testNodes); i++ {
		testEdges = append(testEdges, model.Edge{
			SourceID:     testNodes[i-1].ID,
			TargetID:     testNodes[i].ID,
			RelationType: "communicates_with",
			Weight:       0.8,
			CreatedAt:    now,
		})
	}
	_, _ = ms.InsertEdges(ctx, testEdges)

	const queryCount = 100
	var totalPayloadBaseline int
	var totalPayloadStripped int
	var maxStrippedPayload int
	var minStrippedPayload = 999999999

	for i := 0; i < queryCount; i++ {
		queryTopic := fmt.Sprintf("Entity #%d Buffer", (i%25)+1)

		// 1. Query with include_embeddings=true (baseline representation: ~19–21 KB for 18 nodes)
		reqTrue := model.RecallRequest{
			Query:             queryTopic,
			TopK:              18,
			IncludeEmbeddings: true,
		}
		bodyTrue, _ := json.Marshal(reqTrue)
		httpReqTrue := httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(bodyTrue))
		recTrue := httptest.NewRecorder()
		server.ServeHTTP(recTrue, httpReqTrue)
		if recTrue.Code != http.StatusOK {
			t.Fatalf("baseline recall query %d failed: %d", i, recTrue.Code)
		}
		sizeBaseline := recTrue.Body.Len()
		totalPayloadBaseline += sizeBaseline

		// 2. Query with include_embeddings=false (default: raw float vectors stripped, < 2 KB)
		reqFalse := model.RecallRequest{
			Query: queryTopic,
			TopK:  5, // standard default top-k
		}
		bodyFalse, _ := json.Marshal(reqFalse)
		httpReqFalse := httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(bodyFalse))
		recFalse := httptest.NewRecorder()
		server.ServeHTTP(recFalse, httpReqFalse)
		if recFalse.Code != http.StatusOK {
			t.Fatalf("stripped recall query %d failed: %d", i, recFalse.Code)
		}
		sizeStripped := recFalse.Body.Len()
		totalPayloadStripped += sizeStripped

		if sizeStripped > maxStrippedPayload {
			maxStrippedPayload = sizeStripped
		}
		if sizeStripped < minStrippedPayload {
			minStrippedPayload = sizeStripped
		}

		// Individual query assertion: response payload size MUST be < 2 KB (2048 bytes)
		if sizeStripped >= 2048 {
			t.Fatalf("query %d: stripped response payload size %d bytes exceeded 2 KB threshold. Body:\n%s", i, sizeStripped, recFalse.Body.String())
		}

		// 3. Query with field projection schema (fields=["id", "label", "summary", "score"])
		reqProj := model.RecallRequest{
			Query:  queryTopic,
			TopK:   10,
			Fields: []string{"id", "label", "summary", "score"},
		}
		bodyProj, _ := json.Marshal(reqProj)
		httpReqProj := httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader(bodyProj))
		recProj := httptest.NewRecorder()
		server.ServeHTTP(recProj, httpReqProj)
		if recProj.Code != http.StatusOK {
			t.Fatalf("projected recall query %d failed: %d", i, recProj.Code)
		}
		sizeProj := recProj.Body.Len()
		if sizeProj >= 2048 {
			t.Fatalf("query %d: projected response payload size %d bytes exceeded 2 KB threshold", i, sizeProj)
		}
	}

	avgBaselineKB := float64(totalPayloadBaseline) / float64(queryCount) / 1024.0
	avgStrippedKB := float64(totalPayloadStripped) / float64(queryCount) / 1024.0

	t.Logf("================================================================")
	t.Logf("    RECALL PAYLOAD BENCHMARK (100 QUERIES EMPIRICAL AUDIT)      ")
	t.Logf("================================================================")
	t.Logf(" Baseline Payload (include_embeddings=true):  %.2f KB avg       ", avgBaselineKB)
	t.Logf(" Stripped Payload (include_embeddings=false): %.2f KB avg       ", avgStrippedKB)
	t.Logf(" Min Stripped Payload:                        %d bytes          ", minStrippedPayload)
	t.Logf(" Max Stripped Payload:                        %d bytes (< 2 KB) ", maxStrippedPayload)
	t.Logf(" Total Payload Reduction:                     %.1f%%            ", (1.0-(avgStrippedKB/avgBaselineKB))*100.0)
	t.Logf("================================================================")

	// Verification 1: Baseline payload is ~19-21 KB
	if avgBaselineKB < 18.0 || avgBaselineKB > 22.0 {
		t.Fatalf("expected baseline payload to be ~19-21 KB, got %.2f KB", avgBaselineKB)
	}

	// Verification 2: Stripped payload is strictly < 2 KB
	if avgStrippedKB >= 2.0 {
		t.Fatalf("expected stripped payload to be < 2 KB, got %.2f KB", avgStrippedKB)
	}
}

func TestAPIHealth_EmbeddingEngineTelemetry(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()

	// 1. Standalone mode (no environment configuration, embedding disabled)
	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("failed creating engine: %v", err)
	}

	server := NewServer(ms, engine, 8084)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/memory/health", nil)
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var health model.HealthResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &health); err != nil {
		t.Fatalf("failed unmarshaling health response: %v", err)
	}

	if health.EmbeddingEngine == nil {
		t.Fatalf("expected embedding_engine field in health response, but got nil")
	}

	if health.EmbeddingEngine.Enabled {
		t.Fatalf("expected embedding_engine.enabled = false in standalone mode")
	}
	if health.EmbeddingEngine.Status != "offline" {
		t.Fatalf("expected embedding_engine.status = offline, got %s", health.EmbeddingEngine.Status)
	}

	// 2. Enabled mode (configured with embedder)
	mockEmb := embedding.NewMockEmbedder(384)
	server.SetEmbedder(mockEmb)

	rec2 := httptest.NewRecorder()
	server.ServeHTTP(rec2, req)
	if err := json.Unmarshal(rec2.Body.Bytes(), &health); err != nil {
		t.Fatalf("failed unmarshaling health response: %v", err)
	}

	if !health.EmbeddingEngine.Enabled {
		t.Fatalf("expected embedding_engine.enabled = true when configured")
	}
	if health.EmbeddingEngine.Dimension != 384 {
		t.Fatalf("expected embedding dimension 384, got %d", health.EmbeddingEngine.Dimension)
	}
	if health.EmbeddingEngine.Status != "reachable" {
		t.Fatalf("expected embedding_engine.status = reachable, got %s", health.EmbeddingEngine.Status)
	}
	t.Logf("Embedding telemetry: %+v", *health.EmbeddingEngine)
}

func TestAPIInsert_OnTheFlyDenseEmbedding(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()
	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("failed creating engine: %v", err)
	}

	server := NewServer(ms, engine, 8084)

	// Insert node with OMITTED embedding
	nodeWithoutEmb := model.Node{
		ID:         "node-no-emb",
		EntityType: "concept",
		Label:      "Edge Embedding Engine",
		Summary:    "all-MiniLM-L6-v2 384-dimensional dense semantic vectors",
	}

	body, _ := json.Marshal(model.InsertRequest{
		Nodes: []model.Node{nodeWithoutEmb},
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/insert", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	// Verify node in store received computed 384-D embedding
	storedNode, err := ms.GetNode(ctx, "node-no-emb")
	if err != nil || storedNode == nil {
		t.Fatalf("expected stored node, got error: %v", err)
	}

	if len(storedNode.Embedding) != 384 {
		t.Fatalf("expected 384-D dense embedding to be computed on-the-fly, got %d", len(storedNode.Embedding))
	}
}

func TestAPIRecall_HybridParameters(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()
	now := time.Now().UTC()

	node := model.Node{
		ID:             "hybrid-api-node",
		EntityType:     "concept",
		Label:          "Consensus Cluster Coordination",
		Summary:        "heartbeat timeout configured to 500ms",
		CreatedAt:      now,
		LastAccessedAt: now,
		AccessCount:    1,
		StabilityScore: 1.0,
	}
	_, _ = ms.InsertNodes(ctx, []model.Node{node})

	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("failed creating engine: %v", err)
	}

	server := NewServer(ms, engine, 8084)

	// Test URL query parameter overrides: ?hybrid_alpha=0.90&mode=dense
	body, _ := json.Marshal(model.RecallRequest{
		Query: "consensus heartbeat",
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall?hybrid_alpha=0.90&mode=dense", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp model.RecallResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed unmarshaling recall response: %v", err)
	}

	if len(resp.Nodes) == 0 {
		t.Fatalf("expected recall results")
	}

	// In dense mode, sim_score should equal dense_score
	node0 := resp.Nodes[0]
	if node0.DenseScore == nil || math.Abs(node0.SimScore-*node0.DenseScore) > 1e-4 {
		t.Fatalf("expected sim_score (%f) to equal dense_score (%v)", node0.SimScore, node0.DenseScore)
	}
}

func TestAPIRecall_GatedNeighbourExpansionQueryParameters(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()
	now := time.Now().UTC()

	seedNode := model.Node{
		ID:             "api-seed",
		EntityType:     "concept",
		Label:          "API Primary Seed Node",
		Summary:        "Primary seed entity for recall",
		Embedding:      []float32{1.0, 0.0},
		CreatedAt:      now,
		LastAccessedAt: now,
		AccessCount:    50,
		StabilityScore: 1.0,
	}
	weakNeighbour := model.Node{
		ID:             "api-weak-neighbour",
		EntityType:     "concept",
		Label:          "API Weak Neighbour Node",
		Summary:        "Connected via sub-threshold weight edge",
		Embedding:      []float32{0.0, 1.0},
		CreatedAt:      now,
		LastAccessedAt: now,
		AccessCount:    2,
		StabilityScore: 1.0,
	}
	strongNeighbour := model.Node{
		ID:             "api-strong-neighbour",
		EntityType:     "concept",
		Label:          "API Strong Neighbour Node",
		Summary:        "Connected via valid edge",
		Embedding:      []float32{0.0, 1.0},
		CreatedAt:      now,
		LastAccessedAt: now,
		AccessCount:    2,
		StabilityScore: 1.0,
	}
	subgoalNeighbour := model.Node{
		ID:             "api-subgoal-neighbour",
		EntityType:     "concept",
		Label:          "API Subgoal Neighbour Node",
		Summary:        "Connected via non-whitelisted relation",
		Embedding:      []float32{0.0, 1.0},
		CreatedAt:      now,
		LastAccessedAt: now,
		AccessCount:    2,
		StabilityScore: 1.0,
	}

	_, _ = ms.InsertNodes(ctx, []model.Node{seedNode, weakNeighbour, strongNeighbour, subgoalNeighbour})
	_, _ = ms.InsertEdges(ctx, []model.Edge{
		{SourceID: "api-seed", TargetID: "api-weak-neighbour", RelationType: "depends_on", Weight: 0.45, CreatedAt: now},
		{SourceID: "api-seed", TargetID: "api-strong-neighbour", RelationType: "depends_on", Weight: 0.85, CreatedAt: now},
		{SourceID: "api-seed", TargetID: "api-subgoal-neighbour", RelationType: "subgoal_of", Weight: 0.90, CreatedAt: now},
	})

	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("failed creating engine: %v", err)
	}

	server := NewServer(ms, engine, 8084)

	// 1. Test query param overrides: expand_neighbours=true, min_edge_weight=0.60, traverse_relations=depends_on, attenuation_factor=0.35
	urlStr := "/api/v1/memory/recall?expand_neighbours=true&min_edge_weight=0.60&traverse_relations=depends_on&attenuation_factor=0.35"
	body, _ := json.Marshal(model.RecallRequest{
		Embedding: []float32{1.0, 0.0},
		TopK:      4,
	})
	httpReq := httptest.NewRequest(http.MethodPost, urlStr, bytes.NewReader(body))
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, httpReq)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp model.RecallResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed unmarshaling recall response: %v", err)
	}

	// Verify only the strong neighbour (depends_on with weight 0.85) was traversed
	if len(resp.Edges) != 1 {
		t.Fatalf("expected 1 edge in subgraph, got %d", len(resp.Edges))
	}
	if resp.Edges[0].TargetID != "api-strong-neighbour" {
		t.Fatalf("expected edge to api-strong-neighbour, got %+v", resp.Edges[0])
	}

	// Verify hop_distance is preserved in ScoredNodeDTO JSON
	foundStrong := false
	for _, n := range resp.Nodes {
		if n.ID == "api-strong-neighbour" {
			foundStrong = true
			if n.HopDistance != 1 {
				t.Fatalf("expected hop_distance 1 on strong neighbour, got %d", n.HopDistance)
			}
		}
		if n.ID == "api-weak-neighbour" && n.HopDistance != 0 {
			t.Fatalf("expected hop_distance 0 on untraversed weak neighbour, got %d", n.HopDistance)
		}
	}
	if !foundStrong {
		t.Fatalf("expected api-strong-neighbour to be recalled")
	}

	// Also verify raw JSON contains "hop_distance": 1
	var rawJSON map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &rawJSON); err != nil {
		t.Fatalf("failed unmarshaling raw json: %v", err)
	}
	nodesArr, ok := rawJSON["nodes"].([]any)
	if !ok || len(nodesArr) == 0 {
		t.Fatalf("expected nodes array in raw json")
	}
	hasHopDistanceInJSON := false
	for _, item := range nodesArr {
		nodeMap, ok := item.(map[string]any)
		if ok && nodeMap["id"] == "api-strong-neighbour" {
			if hd, exists := nodeMap["hop_distance"]; exists {
				hasHopDistanceInJSON = true
				if hdFloat, ok := hd.(float64); !ok || int(hdFloat) != 1 {
					t.Fatalf("expected hop_distance 1 in JSON, got %v", hd)
				}
			}
		}
	}
	if !hasHopDistanceInJSON {
		t.Fatalf("expected hop_distance field to be preserved in JSON response for 1-hop neighbour")
	}

	// 2. Test field projection schema preserving hop_distance
	projURL := "/api/v1/memory/recall?expand_neighbours=true&fields=id,label,hop_distance,score"
	httpReqProj := httptest.NewRequest(http.MethodPost, projURL, bytes.NewReader(body))
	recProj := httptest.NewRecorder()
	server.ServeHTTP(recProj, httpReqProj)

	if recProj.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", recProj.Code)
	}

	var rawProj map[string]any
	if err := json.Unmarshal(recProj.Body.Bytes(), &rawProj); err != nil {
		t.Fatalf("failed unmarshaling projected json: %v", err)
	}
	projNodes, ok := rawProj["nodes"].([]any)
	if !ok || len(projNodes) == 0 {
		t.Fatalf("expected projected nodes array")
	}
	firstProj := projNodes[0].(map[string]any)
	if _, exists := firstProj["hop_distance"]; !exists {
		t.Fatalf("expected projected node to include hop_distance field")
	}
}

func TestHandleDecayEndpoint(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()

	var receivedReq model.DecayRequest
	ms.applyScopedDecayFn = func(ctx context.Context, req model.DecayRequest, refTime time.Time) (int, int, int, int, error) {
		receivedReq = req
		return 3, 10, 2, 5, nil
	}

	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("engine initialisation failed: %v", err)
	}

	server := NewServer(ms, engine, 8084)

	endpoints := []string{
		"/api/v1/consolidation/decay",
		"/api/v1/memory/decay",
	}

	payload := `{
		"scope": "unanchored",
		"inactivity_grace_period": "0s",
		"decay_half_life": "1h",
		"min_importance_to_retain": 0.80
	}`

	for _, ep := range endpoints {
		httpReq := httptest.NewRequest(http.MethodPost, ep, bytes.NewReader([]byte(payload)))
		httpReq.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		server.ServeHTTP(rec, httpReq)

		if rec.Code != http.StatusOK {
			t.Fatalf("%s returned status %d: %s", ep, rec.Code, rec.Body.String())
		}

		var resp model.DecayResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("failed unmarshaling decay response: %v", err)
		}

		if resp.Status != "success" {
			t.Errorf("expected status success, got %s", resp.Status)
		}
		if resp.Scope != "unanchored" {
			t.Errorf("expected scope unanchored, got %s", resp.Scope)
		}
		if resp.NodesArchived != 10 {
			t.Errorf("expected 10 nodes archived, got %d", resp.NodesArchived)
		}
		if resp.NodesDecayed != 3 {
			t.Errorf("expected 3 nodes decayed, got %d", resp.NodesDecayed)
		}
		if resp.EdgesPruned != 2 {
			t.Errorf("expected 2 edges pruned, got %d", resp.EdgesPruned)
		}
		if resp.ProtectedNodes != 5 {
			t.Errorf("expected 5 protected nodes, got %d", resp.ProtectedNodes)
		}

		if receivedReq.Scope != "unanchored" {
			t.Errorf("expected received scope unanchored, got %s", receivedReq.Scope)
		}
		if receivedReq.InactivityGracePeriod == nil || *receivedReq.InactivityGracePeriod != 0 {
			t.Errorf("expected 0s grace period, got %v", receivedReq.InactivityGracePeriod)
		}
		if receivedReq.DecayHalfLife == nil || *receivedReq.DecayHalfLife != time.Hour {
			t.Errorf("expected 1h decay half life, got %v", receivedReq.DecayHalfLife)
		}
	}
}

func TestAPI_AuthMiddleware_Enforcement(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()
	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	server := NewServer(ms, engine, 8084)
	server.SetAPIKey("secret123")

	// Protected endpoints to test
	protectedEndpoints := []struct {
		method string
		path   string
		body   string
	}{
		{http.MethodPost, "/api/v1/memory/recall", `{"query": "test"}`},
		{http.MethodPost, "/api/v1/memory/insert", `{"nodes": [], "edges": []}`},
		{http.MethodPost, "/api/v1/memory/consolidate", `{"task_goal": "test", "session_id": "sess-1", "outcome": "success", "steps": []}`},
		{http.MethodPost, "/api/v1/consolidation/decay", `{"scope": "unanchored"}`},
		{http.MethodPost, "/api/v1/memory/decay", `{"scope": "unanchored"}`},
	}

	for _, ep := range protectedEndpoints {
		t.Run("MissingKey_"+ep.path, func(t *testing.T) {
			req := httptest.NewRequest(ep.method, ep.path, bytes.NewReader([]byte(ep.body)))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code != http.StatusUnauthorized {
				t.Errorf("expected 401 Unauthorized for %s, got: %d (%s)", ep.path, rec.Code, rec.Body.String())
			}

			var errResp map[string]string
			if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
				t.Fatalf("failed decoding 401 JSON error: %v", err)
			}
			if errResp["error"] != "unauthorized: invalid or missing API key" {
				t.Errorf("expected error 'unauthorized: invalid or missing API key', got: %q", errResp["error"])
			}
		})

		t.Run("InvalidKey_"+ep.path, func(t *testing.T) {
			req := httptest.NewRequest(ep.method, ep.path, bytes.NewReader([]byte(ep.body)))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-API-Key", "wrong-key")
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code != http.StatusUnauthorized {
				t.Errorf("expected 401 Unauthorized for invalid key on %s, got: %d", ep.path, rec.Code)
			}
		})

		t.Run("InvalidBearer_"+ep.path, func(t *testing.T) {
			req := httptest.NewRequest(ep.method, ep.path, bytes.NewReader([]byte(ep.body)))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer wrong-bearer-token")
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code != http.StatusUnauthorized {
				t.Errorf("expected 401 Unauthorized for invalid bearer on %s, got: %d", ep.path, rec.Code)
			}
		})

		t.Run("ValidXAPIKey_"+ep.path, func(t *testing.T) {
			req := httptest.NewRequest(ep.method, ep.path, bytes.NewReader([]byte(ep.body)))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-API-Key", "secret123")
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code == http.StatusUnauthorized {
				t.Errorf("expected authorized access with valid X-API-Key on %s, got 401", ep.path)
			}
		})

		t.Run("ValidBearer_"+ep.path, func(t *testing.T) {
			req := httptest.NewRequest(ep.method, ep.path, bytes.NewReader([]byte(ep.body)))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer secret123")
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code == http.StatusUnauthorized {
				t.Errorf("expected authorized access with valid Bearer token on %s, got 401", ep.path)
			}
		})
	}

	// Public exemptions without credentials: GET /health and GET /api/v1/memory/health
	publicEndpoints := []string{"/health", "/api/v1/memory/health"}
	for _, pubPath := range publicEndpoints {
		t.Run("PublicExemption_"+pubPath, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, pubPath, nil)
			rec := httptest.NewRecorder()
			server.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Errorf("expected 200 OK for public endpoint %s without credentials, got: %d (%s)", pubPath, rec.Code, rec.Body.String())
			}
		})
	}
}

func TestAPI_AuthMiddleware_PermissiveDevMode(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()
	engine, err := recall.NewEngine(ctx, ms, recall.DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	server := NewServer(ms, engine, 8084)
	server.SetAPIKey("") // permissive / dev mode

	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader([]byte(`{"query": "test"}`)))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200 OK in permissive dev mode when apiKey is empty, got: %d (%s)", rec.Code, rec.Body.String())
	}
}

func TestAPI_Recall_SecretMaskingAndPlaintextRecall(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()

	cipher, err := security.NewAESGCMCipher("api-test-master-key-xyz")
	if err != nil {
		t.Fatalf("failed creating cipher: %v", err)
	}

	cfg := recall.DefaultConfig()
	cfg.Cipher = cipher
	engine, err := recall.NewEngine(ctx, ms, cfg)
	if err != nil {
		t.Fatalf("failed creating recall engine: %v", err)
	}

	server := NewServer(ms, engine, 8084)
	server.SetAPIKey("secret123")

	secretPlain := "API_KEY=hvb_live_999 secret_token"
	encSecret, err := cipher.Encrypt(secretPlain)
	if err != nil {
		t.Fatalf("failed encrypting secret: %v", err)
	}

	vaultRef := "vault://secrets/kestrel-api-key"
	now := time.Now().UTC()

	nodes := []model.Node{
		{
			ID:              "node-sec-api",
			EntityType:      "credential",
			Label:           "HVB Live API Key Node",
			Summary:         encSecret,
			IsSecret:        true,
			Embedding:       embedding.Generate("HVB Live API Key Node", model.DefaultVectorDim),
			CreatedAt:       now,
			LastAccessedAt:  now,
			StabilityScore:  1.0,
			ImportanceScore: 0.9,
		},
		{
			ID:              "node-vault-api",
			EntityType:      "credential",
			Label:           "Kestrel Vault URI Key Node",
			Summary:         vaultRef,
			IsSecret:        true,
			Embedding:       embedding.Generate("Kestrel Vault URI Key Node", model.DefaultVectorDim),
			CreatedAt:       now,
			LastAccessedAt:  now,
			StabilityScore:  1.0,
			ImportanceScore: 0.9,
		},
	}
	_, err = ms.InsertNodes(ctx, nodes)
	if err != nil {
		t.Fatalf("insert nodes failed: %v", err)
	}
	engine.RegisterNodes(nodes)

	// 1. Recall without include_secrets (default) -> summary must be [REDACTED_SECRET]
	reqRedacted := httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader([]byte(`{
		"query": "HVB Live API Key Node",
		"top_k": 5,
		"include_secrets": false
	}`)))
	reqRedacted.Header.Set("Content-Type", "application/json")
	reqRedacted.Header.Set("X-API-Key", "secret123")
	recRedacted := httptest.NewRecorder()
	server.ServeHTTP(recRedacted, reqRedacted)

	if recRedacted.Code != http.StatusOK {
		t.Fatalf("recall failed with status %d: %s", recRedacted.Code, recRedacted.Body.String())
	}

	var respRedacted model.RecallResponse
	if err := json.Unmarshal(recRedacted.Body.Bytes(), &respRedacted); err != nil {
		t.Fatalf("failed unmarshaling recall response: %v", err)
	}

	if len(respRedacted.Nodes) == 0 {
		t.Fatalf("expected recalled nodes, got 0")
	}

	var foundSecret *model.ScoredNode
	for i := range respRedacted.Nodes {
		if respRedacted.Nodes[i].ID == "node-sec-api" {
			foundSecret = &respRedacted.Nodes[i]
			break
		}
	}
	if foundSecret == nil {
		t.Fatalf("node-sec-api not returned in recall")
	}
	if foundSecret.Summary != "[REDACTED_SECRET]" {
		t.Errorf("expected redacted summary '[REDACTED_SECRET]', got: %q", foundSecret.Summary)
	}
	if !foundSecret.IsSecret {
		t.Errorf("expected foundSecret.IsSecret to be true")
	}

	// 2. Recall with include_secrets: true -> decrypted plaintext summary returned
	reqPlain := httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall", bytes.NewReader([]byte(`{
		"query": "HVB Live API Key Node",
		"top_k": 5,
		"include_secrets": true
	}`)))
	reqPlain.Header.Set("Content-Type", "application/json")
	reqPlain.Header.Set("X-API-Key", "secret123")
	recPlain := httptest.NewRecorder()
	server.ServeHTTP(recPlain, reqPlain)

	if recPlain.Code != http.StatusOK {
		t.Fatalf("recall failed with status %d: %s", recPlain.Code, recPlain.Body.String())
	}

	var respPlain model.RecallResponse
	if err := json.Unmarshal(recPlain.Body.Bytes(), &respPlain); err != nil {
		t.Fatalf("failed unmarshaling recall response: %v", err)
	}

	var plainFound *model.ScoredNode
	for i := range respPlain.Nodes {
		if respPlain.Nodes[i].ID == "node-sec-api" {
			plainFound = &respPlain.Nodes[i]
			break
		}
	}
	if plainFound == nil {
		t.Fatalf("node-sec-api not returned in plain recall")
	}
	if plainFound.Summary != secretPlain {
		t.Errorf("expected decrypted summary %q, got: %q", secretPlain, plainFound.Summary)
	}

	// 3. Vault URI node: include_secrets: false masks as [REDACTED_SECRET], include_secrets: true returns vault://
	reqVaultRedacted := httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall?include_secrets=false", bytes.NewReader([]byte(`{
		"query": "Kestrel Vault URI Key Node",
		"top_k": 5
	}`)))
	reqVaultRedacted.Header.Set("Content-Type", "application/json")
	reqVaultRedacted.Header.Set("X-API-Key", "secret123")
	recVaultRedacted := httptest.NewRecorder()
	server.ServeHTTP(recVaultRedacted, reqVaultRedacted)

	var respVR model.RecallResponse
	_ = json.Unmarshal(recVaultRedacted.Body.Bytes(), &respVR)
	for _, n := range respVR.Nodes {
		if n.ID == "node-vault-api" {
			if n.Summary != "[REDACTED_SECRET]" {
				t.Errorf("expected vault reference to be masked when include_secrets=false, got: %q", n.Summary)
			}
		}
	}

	reqVaultPlain := httptest.NewRequest(http.MethodPost, "/api/v1/memory/recall?include_secrets=true", bytes.NewReader([]byte(`{
		"query": "Kestrel Vault URI Key Node",
		"top_k": 5
	}`)))
	reqVaultPlain.Header.Set("Content-Type", "application/json")
	reqVaultPlain.Header.Set("X-API-Key", "secret123")
	recVaultPlain := httptest.NewRecorder()
	server.ServeHTTP(recVaultPlain, reqVaultPlain)

	var respVP model.RecallResponse
	_ = json.Unmarshal(recVaultPlain.Body.Bytes(), &respVP)
	for _, n := range respVP.Nodes {
		if n.ID == "node-vault-api" {
			if n.Summary != vaultRef {
				t.Errorf("expected vault reference %q preserved when include_secrets=true, got: %q", vaultRef, n.Summary)
			}
		}
	}
}

func TestConsolidation_AuthMiddleware_Enforcement(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/v1/consolidation/trigger", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "triggered"})
	})
	mux.HandleFunc("POST /api/v1/consolidation/decay", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "decayed"})
	})
	mux.HandleFunc("GET /api/v1/consolidation/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "healthy", "service": "sekha-consolidation"})
	})
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
	})

	handler := AuthMiddleware("consolidation-secret-key", "/health", "/api/v1/consolidation/health")(mux)

	// 1. Unauthenticated trigger -> 401
	req := httptest.NewRequest(http.MethodPost, "/api/v1/consolidation/trigger", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for unauthenticated trigger, got %d", rec.Code)
	}

	// 2. Unauthenticated decay -> 401
	reqDecay := httptest.NewRequest(http.MethodPost, "/api/v1/consolidation/decay", bytes.NewReader([]byte(`{}`)))
	recDecay := httptest.NewRecorder()
	handler.ServeHTTP(recDecay, reqDecay)
	if recDecay.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for unauthenticated decay, got %d", recDecay.Code)
	}

	// 3. Authenticated trigger with X-API-Key -> 200
	reqAuth := httptest.NewRequest(http.MethodPost, "/api/v1/consolidation/trigger", nil)
	reqAuth.Header.Set("X-API-Key", "consolidation-secret-key")
	recAuth := httptest.NewRecorder()
	handler.ServeHTTP(recAuth, reqAuth)
	if recAuth.Code != http.StatusOK {
		t.Errorf("expected 200 for authenticated trigger, got %d", recAuth.Code)
	}

	// 4. Authenticated trigger with Bearer -> 200
	reqBearer := httptest.NewRequest(http.MethodPost, "/api/v1/consolidation/trigger", nil)
	reqBearer.Header.Set("Authorization", "Bearer consolidation-secret-key")
	recBearer := httptest.NewRecorder()
	handler.ServeHTTP(recBearer, reqBearer)
	if recBearer.Code != http.StatusOK {
		t.Errorf("expected 200 for authenticated trigger with Bearer, got %d", recBearer.Code)
	}

	// 5. Health endpoints without credentials -> 200
	for _, hp := range []string{"/health", "/api/v1/consolidation/health"} {
		reqH := httptest.NewRequest(http.MethodGet, hp, nil)
		recH := httptest.NewRecorder()
		handler.ServeHTTP(recH, reqH)
		if recH.Code != http.StatusOK {
			t.Errorf("expected 200 for public health endpoint %s, got %d", hp, recH.Code)
		}
	}
}

func TestAPI_ConsolidateSecretFlagPropagation(t *testing.T) {
	ctx := context.Background()
	ms := newAPIMockStore()

	cipher, err := security.NewAESGCMCipher("")
	if err != nil {
		t.Fatalf("failed creating cipher: %v", err)
	}

	cfg := recall.DefaultConfig()
	cfg.Cipher = cipher
	engine, err := recall.NewEngine(ctx, ms, cfg)
	if err != nil {
		t.Fatalf("failed creating recall engine: %v", err)
	}

	server := NewServer(ms, engine, 8084)
	server.SetAPIKey("secret123")

	payload := model.ConsolidateRequest{
		TraceID:     "trace-api-sec-01",
		SessionID:   "sess-api-sec",
		TaskGoal:    "Provision Kestrel Master Token",
		Outcome:     model.OutcomeSuccess,
		Synchronous: true,
		IsSecret:    true,
		Trajectory: []model.TrajectoryStep{
			{
				StepIndex:   1,
				Thought:     "Generating cluster secret token-12345",
				Action:      "generate_token",
				Observation: "Token generated",
				Status:      "success",
			},
		},
	}

	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/consolidate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", "secret123")
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp model.ConsolidateResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed unmarshaling consolidate response: %v", err)
	}

	if len(resp.CreatedNodes) == 0 {
		t.Fatalf("expected created nodes, got 0")
	}

	for _, n := range resp.CreatedNodes {
		if !n.IsSecret {
			t.Errorf("expected created node %s to carry IsSecret = true", n.ID)
		}
	}
}


