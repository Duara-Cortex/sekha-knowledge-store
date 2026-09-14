package recall

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

// mockStore provides an in-memory Store for unit testing the recall engine.
type mockStore struct {
	nodes   map[string]model.Node
	edges   []model.Edge
	headers []store.NodeHeader
}

func newMockStore() *mockStore {
	return &mockStore{
		nodes: make(map[string]model.Node),
	}
}

func (m *mockStore) Close() error { return nil }

func (m *mockStore) InsertNodes(ctx context.Context, nodes []model.Node) (int, error) {
	for _, n := range nodes {
		m.nodes[n.ID] = n
		var mag float32
		if len(n.Embedding) > 0 {
			var sum float64
			for _, v := range n.Embedding {
				sum += float64(v * v)
			}
			mag = float32(math.Sqrt(sum))
		}
		m.headers = append(m.headers, store.NodeHeader{
			ID:             n.ID,
			EntityType:     n.EntityType,
			Embedding:      n.Embedding,
			Magnitude:      mag,
			LastAccessedAt: n.LastAccessedAt,
			AccessCount:    n.AccessCount,
			StabilityScore: n.StabilityScore,
		})
	}
	return len(nodes), nil
}

func (m *mockStore) InsertEdges(ctx context.Context, edges []model.Edge) (int, error) {
	m.edges = append(m.edges, edges...)
	return len(edges), nil
}

func (m *mockStore) GetNode(ctx context.Context, id string) (*model.Node, error) {
	n, exists := m.nodes[id]
	if !exists {
		return nil, nil
	}
	return &n, nil
}

func (m *mockStore) GetNodes(ctx context.Context, ids []string) (map[string]model.Node, error) {
	res := make(map[string]model.Node)
	for _, id := range ids {
		if n, ok := m.nodes[id]; ok {
			res[id] = n
		}
	}
	return res, nil
}

func (m *mockStore) GetAllNodeHeaders(ctx context.Context) ([]store.NodeHeader, error) {
	return m.headers, nil
}

func (m *mockStore) GetEdgesForNodes(ctx context.Context, nodeIDs []string) ([]model.Edge, error) {
	set := make(map[string]bool)
	for _, id := range nodeIDs {
		set[id] = true
	}
	var res []model.Edge
	for _, e := range m.edges {
		if set[e.SourceID] || set[e.TargetID] {
			res = append(res, e)
		}
	}
	return res, nil
}

func (m *mockStore) RecordAccess(ctx context.Context, nodeIDs []string, accessTime time.Time) error {
	for _, id := range nodeIDs {
		if n, ok := m.nodes[id]; ok {
			n.AccessCount++
			n.LastAccessedAt = accessTime
			m.nodes[id] = n
		}
	}
	return nil
}

func (m *mockStore) GetCounts(ctx context.Context) (int64, int64, error) {
	return int64(len(m.nodes)), int64(len(m.edges)), nil
}

func (m *mockStore) GetGraphSummary(ctx context.Context) (*model.GraphSummary, error) {
	return &model.GraphSummary{
		NodeCount: int64(len(m.nodes)),
		EdgeCount: int64(len(m.edges)),
	}, nil
}

func TestRecallEngine_WeightedRanking(t *testing.T) {
	ctx := context.Background()
	ms := newMockStore()

	now := time.Now().UTC()
	// Node 1: High semantic similarity, low access count, recent
	n1 := model.Node{
		ID:             "node-1",
		EntityType:     "concept",
		Label:          "ARM NEON Vectorization",
		Summary:        "SIMD acceleration on ARM Cortex-A76",
		Embedding:      []float32{1.0, 0.0, 0.0},
		CreatedAt:      now.Add(-24 * time.Hour),
		LastAccessedAt: now.Add(-5 * time.Minute),
		AccessCount:    2,
		StabilityScore: 1.0,
	}

	// Node 2: Low semantic similarity, very high access count (frequently used concept)
	n2 := model.Node{
		ID:             "node-2",
		EntityType:     "concept",
		Label:          "Cluster Switch Architecture",
		Summary:        "Gigabit switch linking nodes",
		Embedding:      []float32{0.0, 1.0, 0.0},
		CreatedAt:      now.Add(-48 * time.Hour),
		LastAccessedAt: now.Add(-10 * time.Minute),
		AccessCount:    80,
		StabilityScore: 1.0,
	}

	// Node 3: Old node, not accessed recently
	n3 := model.Node{
		ID:             "node-3",
		EntityType:     "concept",
		Label:          "Obsolete Boot Config",
		Summary:        "Legacy systemd script",
		Embedding:      []float32{0.5, 0.5, 0.0},
		CreatedAt:      now.Add(-720 * time.Hour),
		LastAccessedAt: now.Add(-500 * time.Hour),
		AccessCount:    1,
		StabilityScore: 0.5,
	}

	_, _ = ms.InsertNodes(ctx, []model.Node{n1, n2, n3})

	engine, err := NewEngine(ctx, ms, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	// Query matching node-1 embedding
	req := model.RecallRequest{
		Embedding:  []float32{1.0, 0.0, 0.0},
		TopK:       3,
		Alpha:      0.6,
		Beta:       0.2,
		Gamma:      0.2,
		ExpandHops: 0,
	}

	resp, err := engine.Recall(ctx, req)
	if err != nil {
		t.Fatalf("recall error: %v", err)
	}

	if len(resp.Nodes) != 3 {
		t.Fatalf("expected 3 returned nodes, got %d", len(resp.Nodes))
	}

	// Node 1 should rank first due to high similarity (alpha = 0.6)
	if resp.Nodes[0].ID != "node-1" {
		t.Fatalf("expected top node to be node-1, got %s", resp.Nodes[0].ID)
	}

	if resp.Nodes[0].SimScore <= resp.Nodes[1].SimScore {
		t.Fatalf("expected node-1 similarity > node-2 similarity")
	}

	t.Logf("Rank 1: %s (score: %f, sim: %f, freq: %f, recency: %f)",
		resp.Nodes[0].ID, resp.Nodes[0].Score, resp.Nodes[0].SimScore, resp.Nodes[0].FrequencyScore, resp.Nodes[0].RecencyScore)
	t.Logf("Rank 2: %s (score: %f, sim: %f, freq: %f, recency: %f)",
		resp.Nodes[1].ID, resp.Nodes[1].Score, resp.Nodes[1].SimScore, resp.Nodes[1].FrequencyScore, resp.Nodes[1].RecencyScore)
	t.Logf("Rank 3: %s (score: %f, sim: %f, freq: %f, recency: %f)",
		resp.Nodes[2].ID, resp.Nodes[2].Score, resp.Nodes[2].SimScore, resp.Nodes[2].FrequencyScore, resp.Nodes[2].RecencyScore)
}

func TestRecallEngine_OneHopNeighborExpansion(t *testing.T) {
	ctx := context.Background()
	ms := newMockStore()

	now := time.Now().UTC()
	// Target node A matches vector
	nA := model.Node{
		ID:             "node-A",
		EntityType:     "issue",
		Label:          "Packet drop alert",
		Summary:        "High drop rate detected on switch",
		Embedding:      []float32{0.9, 0.1},
		CreatedAt:      now,
		LastAccessedAt: now,
		AccessCount:    5,
		StabilityScore: 1.0,
	}

	// Neighbor node B has orthogonal vector (0.0 similarity to query), but linked to node A
	nB := model.Node{
		ID:             "node-B",
		EntityType:     "solution",
		Label:          "Inspect Port 2 Buffers",
		Summary:        "Port 2 connects to Node 3",
		Embedding:      []float32{0.0, 1.0},
		CreatedAt:      now,
		LastAccessedAt: now.Add(-1 * time.Hour),
		AccessCount:    2,
		StabilityScore: 1.0,
	}

	// Irrelevant node C
	nC := model.Node{
		ID:             "node-C",
		EntityType:     "concept",
		Label:          "Unrelated Note",
		Summary:        "Nothing to do with packets",
		Embedding:      []float32{0.1, 0.0},
		CreatedAt:      now,
		LastAccessedAt: now.Add(-24 * time.Hour),
		AccessCount:    0,
		StabilityScore: 1.0,
	}

	_, _ = ms.InsertNodes(ctx, []model.Node{nA, nB, nC})
	_, _ = ms.InsertEdges(ctx, []model.Edge{
		{SourceID: "node-A", TargetID: "node-B", RelationType: "suggests_action", Weight: 1.0, CreatedAt: now},
	})

	engine, err := NewEngine(ctx, ms, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	// Recall with 1-hop expansion
	req := model.RecallRequest{
		Embedding:  []float32{0.9, 0.1},
		TopK:       2,
		Alpha:      0.7,
		Beta:       0.1,
		Gamma:      0.2,
		ExpandHops: 1,
	}

	resp, err := engine.Recall(ctx, req)
	if err != nil {
		t.Fatalf("recall failed: %v", err)
	}

	if len(resp.Nodes) != 2 {
		t.Fatalf("expected top 2 nodes, got %d", len(resp.Nodes))
	}

	// Top should be node-A, second should be node-B due to 1-hop link boost
	if resp.Nodes[0].ID != "node-A" {
		t.Fatalf("expected top node node-A, got %s", resp.Nodes[0].ID)
	}
	if resp.Nodes[1].ID != "node-B" {
		t.Fatalf("expected 1-hop neighbor node-B to be boosted into top 2, got %s", resp.Nodes[1].ID)
	}

	// Subgraph edges should include the edge between A and B
	if len(resp.Edges) == 0 {
		t.Fatalf("expected subgraph edges to be returned")
	}
	if resp.Edges[0].RelationType != "suggests_action" {
		t.Fatalf("unexpected edge relation: %s", resp.Edges[0].RelationType)
	}
}
