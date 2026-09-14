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
}

func newAPIMockStore() *apiMockStore {
	return &apiMockStore{
		nodes: make(map[string]model.Node),
	}
}

func (m *apiMockStore) Close() error { return nil }

func (m *apiMockStore) InsertNodes(ctx context.Context, nodes []model.Node) (int, error) {
	for _, n := range nodes {
		m.nodes[n.ID] = n
		m.headers = append(m.headers, store.NodeHeader{
			ID:             n.ID,
			EntityType:     n.EntityType,
			Embedding:      n.Embedding,
			LastAccessedAt: n.LastAccessedAt,
			AccessCount:    n.AccessCount,
			StabilityScore: n.StabilityScore,
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
	return &n, nil
}

func (m *apiMockStore) GetNodes(ctx context.Context, ids []string) (map[string]model.Node, error) {
	res := make(map[string]model.Node)
	for _, id := range ids {
		if n, ok := m.nodes[id]; ok {
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
