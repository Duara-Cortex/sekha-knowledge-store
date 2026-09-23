package recall

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

// mockStore provides an in-memory Store for unit testing the recall engine.
type mockStore struct {
	mu      sync.RWMutex
	nodes   map[string]model.Node
	edges   []model.Edge
	headers []store.NodeHeader
	anchors map[string][]string
}

func newMockStore() *mockStore {
	return &mockStore{
		nodes:   make(map[string]model.Node),
		anchors: make(map[string][]string),
	}
}

func (m *mockStore) Close() error { return nil }

func (m *mockStore) attachAnchorsLocked(nodeID string, anchors []string) {
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
}

func (m *mockStore) AttachAnchors(ctx context.Context, nodeID string, anchors []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.attachAnchorsLocked(nodeID, anchors)
	return nil
}

func (m *mockStore) GetAnchorsForNode(ctx context.Context, nodeID string) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	res := make([]string, len(m.anchors[nodeID]))
	copy(res, m.anchors[nodeID])
	return res, nil
}

func (m *mockStore) GetNodeIDsForAnchors(ctx context.Context, anchors []string) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
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

func (m *mockStore) InsertNodes(ctx context.Context, nodes []model.Node) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, n := range nodes {
		if len(n.Anchors) > 0 {
			m.attachAnchorsLocked(n.ID, n.Anchors)
			n.Anchors = m.anchors[n.ID]
		}
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
			Label:          n.Label,
			Summary:        n.Summary,
			Embedding:      n.Embedding,
			Magnitude:      mag,
			LastAccessedAt: n.LastAccessedAt,
			AccessCount:    n.AccessCount,
			StabilityScore: n.StabilityScore,
			Anchors:        m.anchors[n.ID],
		})
	}
	return len(nodes), nil
}

func (m *mockStore) InsertEdges(ctx context.Context, edges []model.Edge) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.edges = append(m.edges, edges...)
	return len(edges), nil
}

func (m *mockStore) GetNode(ctx context.Context, id string) (*model.Node, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	n, exists := m.nodes[id]
	if !exists {
		return nil, nil
	}
	n.Anchors = m.anchors[id]
	return &n, nil
}

func (m *mockStore) GetNodes(ctx context.Context, ids []string) (map[string]model.Node, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	res := make(map[string]model.Node)
	for _, id := range ids {
		if n, ok := m.nodes[id]; ok {
			n.Anchors = m.anchors[id]
			res[id] = n
		}
	}
	return res, nil
}

func (m *mockStore) GetAllNodeHeaders(ctx context.Context) ([]store.NodeHeader, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	res := make([]store.NodeHeader, len(m.headers))
	copy(res, m.headers)
	return res, nil
}

func (m *mockStore) GetEdgesForNodes(ctx context.Context, nodeIDs []string) ([]model.Edge, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
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
	m.mu.Lock()
	defer m.mu.Unlock()
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
	m.mu.RLock()
	defer m.mu.RUnlock()
	return int64(len(m.nodes)), int64(len(m.edges)), nil
}

func (m *mockStore) GetGraphSummary(ctx context.Context) (*model.GraphSummary, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return &model.GraphSummary{
		NodeCount: int64(len(m.nodes)),
		EdgeCount: int64(len(m.edges)),
	}, nil
}

func (m *mockStore) FindMatchingNode(ctx context.Context, label string, entityType string) (*model.Node, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, n := range m.nodes {
		if n.Label == label {
			return &n, nil
		}
	}
	return nil, nil
}

func (m *mockStore) QueueTrace(ctx context.Context, trace model.EpisodicTrace) error {
	return nil
}

func (m *mockStore) GetPendingTraces(ctx context.Context, limit int) ([]model.EpisodicTrace, error) {
	return nil, nil
}

func (m *mockStore) MarkTraceConsolidated(ctx context.Context, traceID string, consolidatedAt time.Time) error {
	return nil
}

func (m *mockStore) ReinforceEdge(ctx context.Context, sourceID, targetID, relationType string, deltaW float64, maxWeight float64, reinforcedAt time.Time) (float64, error) {
	return 1.0 + deltaW, nil
}

func (m *mockStore) BoostNodeStability(ctx context.Context, nodeID string, deltaStability float64, reinforcedAt time.Time) error {
	return nil
}

func (m *mockStore) ApplyDecayAndPrune(ctx context.Context, cfg model.DecayConfig, refTime time.Time) (decayed int, archived int, prunedEdges int, err error) {
	return 0, 0, 0, nil
}

func (m *mockStore) GetConsolidationStats(ctx context.Context) (*model.ConsolidationStats, error) {
	return &model.ConsolidationStats{
		ActiveNodes: int64(len(m.nodes)),
		ActiveEdges: int64(len(m.edges)),
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

func TestRecallEngine_OneHopNeighbourExpansion(t *testing.T) {
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

	// Neighbour node B has orthogonal vector (0.0 similarity to query), but linked to node A
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

	// Irrelevant node C (orthogonal embedding to query)
	nC := model.Node{
		ID:             "node-C",
		EntityType:     "concept",
		Label:          "Unrelated Note",
		Summary:        "Nothing to do with packets",
		Embedding:      []float32{-0.9, 0.1},
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
		t.Fatalf("expected 1-hop neighbour node-B to be boosted into top 2, got %s", resp.Nodes[1].ID)
	}

	// Subgraph edges should include the edge between A and B
	if len(resp.Edges) == 0 {
		t.Fatalf("expected subgraph edges to be returned")
	}
	if resp.Edges[0].RelationType != "suggests_action" {
		t.Fatalf("unexpected edge relation: %s", resp.Edges[0].RelationType)
	}
}

func TestRecallEngine_QueryTextSemanticRetrieval(t *testing.T) {
	ctx := context.Background()
	ms := newMockStore()

	now := time.Now().UTC()
	// Node 1: Project Kestrel settings
	n1 := model.Node{
		ID:             "needle-kestrel",
		EntityType:     "decision",
		Label:          "Project Kestrel ingest service settings",
		Summary:        "Configured ingest service parameters and port bindings for Project Kestrel",
		CreatedAt:      now.Add(-48 * time.Hour),
		LastAccessedAt: now.Add(-48 * time.Hour),
		AccessCount:    1,
		StabilityScore: 1.0,
	}

	// Node 2: Unrelated recent node
	n2 := model.Node{
		ID:             "distractor-recent",
		EntityType:     "telemetry",
		Label:          "Camera frame dropped",
		Summary:        "Optical sensor buffer overflow on video stream",
		CreatedAt:      now.Add(-1 * time.Minute),
		LastAccessedAt: now.Add(-1 * time.Minute),
		AccessCount:    50,
		StabilityScore: 1.0,
	}

	_, _ = ms.InsertNodes(ctx, []model.Node{n1, n2})

	engine, err := NewEngine(ctx, ms, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	// Query with text string ONLY (no precomputed embedding)
	req := model.RecallRequest{
		Query:      "Project Kestrel ingest service settings",
		TopK:       2,
		Alpha:      0.6,
		Beta:       0.2,
		Gamma:      0.2,
		ExpandHops: 0,
	}

	resp, err := engine.Recall(ctx, req)
	if err != nil {
		t.Fatalf("recall with query text failed: %v", err)
	}

	if len(resp.Nodes) == 0 {
		t.Fatalf("expected returned nodes, got 0")
	}

	// The needle should rank #1 despite being 48h older and having only 1 access
	if resp.Nodes[0].ID != "needle-kestrel" {
		t.Fatalf("expected needle-kestrel to rank first, got %s (score: %f vs %f)",
			resp.Nodes[0].ID, resp.Nodes[0].Score, resp.Nodes[1].Score)
	}

	if resp.Nodes[0].SimScore <= 0.7 {
		t.Fatalf("expected high SimScore for semantic text match, got %f", resp.Nodes[0].SimScore)
	}

	t.Logf("Rank 1: %s (sim: %f, score: %f)", resp.Nodes[0].ID, resp.Nodes[0].SimScore, resp.Nodes[0].Score)
	t.Logf("Rank 2: %s (sim: %f, score: %f)", resp.Nodes[1].ID, resp.Nodes[1].SimScore, resp.Nodes[1].Score)
}

func TestRecallEngine_EmptyIndexHydration(t *testing.T) {
	ctx := context.Background()
	ms := newMockStore()

	// Engine starts with 0 nodes in store
	engine, err := NewEngine(ctx, ms, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	if engine.IndexLen() != 0 {
		t.Fatalf("expected initial index length 0, got %d", engine.IndexLen())
	}

	// Now nodes are inserted into SQLite store externally
	now := time.Now().UTC()
	_, _ = ms.InsertNodes(ctx, []model.Node{
		{
			ID:             "external-node-1",
			EntityType:     "concept",
			Label:          "Edge cluster coordination",
			Summary:        "Leader election protocol",
			CreatedAt:      now,
			LastAccessedAt: now,
			AccessCount:    5,
			StabilityScore: 1.0,
		},
	})

	// Query recall: should automatically rehydrate the empty index from store
	req := model.RecallRequest{
		Query: "coordination",
		TopK:  5,
	}

	resp, err := engine.Recall(ctx, req)
	if err != nil {
		t.Fatalf("recall failed: %v", err)
	}

	if len(resp.Nodes) != 1 {
		t.Fatalf("expected 1 node returned after empty index auto-hydration, got %d", len(resp.Nodes))
	}

	if resp.Nodes[0].ID != "external-node-1" {
		t.Fatalf("expected external-node-1, got %s", resp.Nodes[0].ID)
	}
}

func TestRecallEngine_OvercomesRecencyTrap(t *testing.T) {
	ctx := context.Background()
	ms := newMockStore()

	now := time.Now().UTC()
	// Target needle: 7 days old, low access count
	needle := model.Node{
		ID:             "needle-kestrel",
		EntityType:     "decision",
		Label:          "Kestrel Ingest Service Port",
		Summary:        "Default port 8084 assigned to kestrel ingest daemon",
		CreatedAt:      now.Add(-7 * 24 * time.Hour),
		LastAccessedAt: now.Add(-7 * 24 * time.Hour),
		AccessCount:    1,
		StabilityScore: 1.0,
	}

	// 5 recent distractor nodes consolidated afterwards (1 minute ago)
	distractors := make([]model.Node, 5)
	for i := 0; i < 5; i++ {
		distractors[i] = model.Node{
			ID:             fmt.Sprintf("distractor-%d", i+1),
			EntityType:     "telemetry",
			Label:          fmt.Sprintf("Distractor Sensor Event #%d", i+1),
			Summary:        fmt.Sprintf("Routine periodic sensor reading #%d", i+1),
			CreatedAt:      now.Add(-time.Duration(i+1) * time.Minute),
			LastAccessedAt: now.Add(-time.Duration(i+1) * time.Minute),
			AccessCount:    int64(20 + i*10),
			StabilityScore: 1.0,
		}
	}

	allNodes := append([]model.Node{needle}, distractors...)
	_, _ = ms.InsertNodes(ctx, allNodes)

	engine, err := NewEngine(ctx, ms, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	// Recall query for the needle using natural text
	req := model.RecallRequest{
		Query: "kestrel ingest service port",
		TopK:  3,
		Alpha: 0.6,
		Beta:  0.2,
		Gamma: 0.2,
	}

	resp, err := engine.Recall(ctx, req)
	if err != nil {
		t.Fatalf("recall failed: %v", err)
	}

	if len(resp.Nodes) == 0 {
		t.Fatalf("expected results, got empty")
	}

	// Needle MUST be rank 1, overcoming the recency and frequency bias of distractors
	if resp.Nodes[0].ID != "needle-kestrel" {
		t.Fatalf("recency trap occurred: expected needle-kestrel at rank 1, but got %s (score %f vs needle %f)",
			resp.Nodes[0].ID, resp.Nodes[0].Score, resp.Nodes[1].Score)
	}

	t.Logf("Needle score: %f (sim: %f, recency: %f)",
		resp.Nodes[0].Score, resp.Nodes[0].SimScore, resp.Nodes[0].RecencyScore)
	t.Logf("Distractor score: %f (sim: %f, recency: %f)",
		resp.Nodes[1].Score, resp.Nodes[1].SimScore, resp.Nodes[1].RecencyScore)
}

func TestRecallEngine_AnchorModeBoost(t *testing.T) {
	ctx := context.Background()
	ms := newMockStore()

	now := time.Now().UTC()

	// Distractor Node: Project Falcon has higher generic similarity, high access count, and high recency
	distractor := model.Node{
		ID:             "falcon-port-timeout",
		EntityType:     "decision",
		Label:          "Falcon port timeout buffer configuration",
		Summary:        "Port timeout and buffer retry limits for falcon cluster node",
		Anchors:        []string{"#project:falcon"},
		CreatedAt:      now.Add(-10 * time.Minute),
		LastAccessedAt: now.Add(-10 * time.Minute),
		AccessCount:    80,
		StabilityScore: 1.0,
	}

	// Target Node: Project Kestrel has lower generic similarity, low access count, and is older
	target := model.Node{
		ID:             "kestrel-port-timeout",
		EntityType:     "decision",
		Label:          "Port timeout parameters",
		Summary:        "Timeout settings",
		Anchors:        []string{"#project:kestrel"},
		CreatedAt:      now.Add(-72 * time.Hour),
		LastAccessedAt: now.Add(-72 * time.Hour),
		AccessCount:    1,
		StabilityScore: 1.0,
	}

	_, _ = ms.InsertNodes(ctx, []model.Node{distractor, target})

	engine, err := NewEngine(ctx, ms, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	// 1. Without anchors: distractor dominates due to higher lexical similarity, recency, and frequency
	baselineResp, err := engine.Recall(ctx, model.RecallRequest{
		Query: "Falcon port timeout buffer configuration",
		TopK:  2,
	})
	if err != nil {
		t.Fatalf("baseline recall failed: %v", err)
	}
	if len(baselineResp.Nodes) < 2 || baselineResp.Nodes[0].ID != "falcon-port-timeout" {
		t.Fatalf("expected falcon-port-timeout at rank 1 in baseline, got %s", baselineResp.Nodes[0].ID)
	}

	// 2. With #project:kestrel anchor and anchor_mode: "boost"
	boostResp, err := engine.Recall(ctx, model.RecallRequest{
		Query:      "Falcon port timeout buffer configuration",
		Anchors:    []string{"#project:kestrel"},
		AnchorMode: "boost",
		TopK:       2,
	})
	if err != nil {
		t.Fatalf("boost recall failed: %v", err)
	}

	if len(boostResp.Nodes) != 2 {
		t.Fatalf("expected 2 returned nodes in boost mode, got %d", len(boostResp.Nodes))
	}

	// Kestrel MUST be elevated to Rank 1 over the higher generic similarity Falcon node
	if boostResp.Nodes[0].ID != "kestrel-port-timeout" {
		t.Fatalf("expected kestrel-port-timeout elevated to rank 1, but got %s (score: %f vs %f)",
			boostResp.Nodes[0].ID, boostResp.Nodes[0].Score, boostResp.Nodes[1].Score)
	}
	if boostResp.Nodes[0].AnchorScore != 1.0 {
		t.Fatalf("expected anchor_score 1.0 on rank 1 node, got %f", boostResp.Nodes[0].AnchorScore)
	}

	// Non-anchored associative results are NOT excluded: Falcon should still be at Rank 2
	if boostResp.Nodes[1].ID != "falcon-port-timeout" {
		t.Fatalf("expected falcon-port-timeout at rank 2, got %s", boostResp.Nodes[1].ID)
	}
	if boostResp.Nodes[1].AnchorScore != 0.0 {
		t.Fatalf("expected anchor_score 0.0 on unanchored rank 2 node, got %f", boostResp.Nodes[1].AnchorScore)
	}

	t.Logf("Rank 1 (Anchored): %s (Score: %f, Sim: %f, AnchorScore: %f)",
		boostResp.Nodes[0].ID, boostResp.Nodes[0].Score, boostResp.Nodes[0].SimScore, boostResp.Nodes[0].AnchorScore)
	t.Logf("Rank 2 (Unanchored): %s (Score: %f, Sim: %f, AnchorScore: %f)",
		boostResp.Nodes[1].ID, boostResp.Nodes[1].Score, boostResp.Nodes[1].SimScore, boostResp.Nodes[1].AnchorScore)
}

func TestRecallEngine_AnchorModeFilter(t *testing.T) {
	ctx := context.Background()
	ms := newMockStore()

	now := time.Now().UTC()

	nodeKestrel := model.Node{
		ID:             "kestrel-node",
		EntityType:     "decision",
		Label:          "Kestrel Service Port",
		Summary:        "Port 8084 binding",
		Anchors:        []string{"#project:kestrel"},
		CreatedAt:      now,
		LastAccessedAt: now,
		StabilityScore: 1.0,
	}

	nodeFalcon := model.Node{
		ID:             "falcon-node",
		EntityType:     "decision",
		Label:          "Falcon Service Port",
		Summary:        "Port 8084 binding",
		Anchors:        []string{"#project:falcon"},
		CreatedAt:      now,
		LastAccessedAt: now,
		StabilityScore: 1.0,
	}

	nodeUnanchored := model.Node{
		ID:             "unanchored-node",
		EntityType:     "decision",
		Label:          "Generic Service Port",
		Summary:        "Port 8084 binding",
		CreatedAt:      now,
		LastAccessedAt: now,
		StabilityScore: 1.0,
	}

	_, _ = ms.InsertNodes(ctx, []model.Node{nodeKestrel, nodeFalcon, nodeUnanchored})

	engine, err := NewEngine(ctx, ms, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	// Query with anchor_mode: "filter" targeting #project:kestrel
	filterResp, err := engine.Recall(ctx, model.RecallRequest{
		Query:      "service port",
		Anchors:    []string{"#project:kestrel"},
		AnchorMode: "filter",
		TopK:       5,
	})
	if err != nil {
		t.Fatalf("filter recall failed: %v", err)
	}

	// Strictly only matching nodes must be returned
	if len(filterResp.Nodes) != 1 {
		t.Fatalf("expected exactly 1 node returned under filter mode, got %d", len(filterResp.Nodes))
	}
	if filterResp.Nodes[0].ID != "kestrel-node" {
		t.Fatalf("expected kestrel-node, got %s", filterResp.Nodes[0].ID)
	}

	// Query with non-matching anchor: must return 0 results
	nonMatchingResp, err := engine.Recall(ctx, model.RecallRequest{
		Query:      "service port",
		Anchors:    []string{"#project:nonexistent"},
		AnchorMode: "filter",
		TopK:       5,
	})
	if err != nil {
		t.Fatalf("filter recall failed: %v", err)
	}
	if len(nonMatchingResp.Nodes) != 0 {
		t.Fatalf("expected 0 results for non-matching filter, got %d", len(nonMatchingResp.Nodes))
	}
}

func TestRecallEngine_AnchorsOmittedRegression(t *testing.T) {
	ctx := context.Background()
	ms := newMockStore()

	now := time.Now().UTC()

	node1 := model.Node{
		ID:             "dominant-node",
		EntityType:     "concept",
		Label:          "Database connection timeout",
		Summary:        "Connection timeout after 30 seconds",
		Anchors:        []string{"#project:alpha"},
		CreatedAt:      now,
		LastAccessedAt: now,
		AccessCount:    50,
		StabilityScore: 1.0,
	}

	node2 := model.Node{
		ID:             "secondary-node",
		EntityType:     "concept",
		Label:          "Socket timeout",
		Summary:        "Low level socket timeout",
		Anchors:        []string{"#project:beta"},
		CreatedAt:      now.Add(-48 * time.Hour),
		LastAccessedAt: now.Add(-48 * time.Hour),
		AccessCount:    2,
		StabilityScore: 1.0,
	}

	_, _ = ms.InsertNodes(ctx, []model.Node{node1, node2})

	engine, err := NewEngine(ctx, ms, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	// Query with empty anchors: queries must execute globally across the entire graph without anchor bias
	resp, err := engine.Recall(ctx, model.RecallRequest{
		Query: "database connection timeout",
		TopK:  2,
	})
	if err != nil {
		t.Fatalf("recall failed: %v", err)
	}

	if len(resp.Nodes) != 2 {
		t.Fatalf("expected 2 nodes returned, got %d", len(resp.Nodes))
	}
	if resp.Nodes[0].ID != "dominant-node" {
		t.Fatalf("expected dominant-node at rank 1, got %s", resp.Nodes[0].ID)
	}
	if resp.Nodes[0].AnchorScore != 0.0 || resp.Nodes[1].AnchorScore != 0.0 {
		t.Fatalf("expected 0.0 anchor scores when anchors omitted, got %f and %f",
			resp.Nodes[0].AnchorScore, resp.Nodes[1].AnchorScore)
	}
}

func TestRecallEngine_AcceptanceCriterion1_SemanticParaphrasing(t *testing.T) {
	ctx := context.Background()
	ms := newMockStore()
	now := time.Now().UTC()

	// Target node using technical terminology "consensus heartbeat timeout"
	targetNode := model.Node{
		ID:             "target-consensus-heartbeat",
		EntityType:     "decision",
		Label:          "Consensus Cluster Configuration",
		Summary:        "consensus heartbeat timeout configured to 500ms for node health monitoring",
		CreatedAt:      now.Add(-48 * time.Hour),
		LastAccessedAt: now.Add(-48 * time.Hour),
		AccessCount:    2,
		StabilityScore: 1.0,
	}

	// Multiple recent distractor nodes
	distractors := []model.Node{
		{
			ID:             "distractor-database",
			EntityType:     "concept",
			Label:          "Database WAL Maintenance",
			Summary:        "sqlite checkpoint routine and journal mode settings",
			CreatedAt:      now.Add(-5 * time.Minute),
			LastAccessedAt: now.Add(-5 * time.Minute),
			AccessCount:    25,
			StabilityScore: 1.0,
		},
		{
			ID:             "distractor-thermal",
			EntityType:     "telemetry",
			Label:          "Cryogenic Refrigerator Sensor",
			Summary:        "thermal sensor telemetry logging threshold alarms",
			CreatedAt:      now.Add(-1 * time.Minute),
			LastAccessedAt: now.Add(-1 * time.Minute),
			AccessCount:    40,
			StabilityScore: 1.0,
		},
	}

	allNodes := append([]model.Node{targetNode}, distractors...)
	_, _ = ms.InsertNodes(ctx, allNodes)

	engine, err := NewEngine(ctx, ms, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	// Conceptual paraphrase query: "cluster coordination interval"
	req := model.RecallRequest{
		Query: "cluster coordination interval",
		TopK:  3,
		Alpha: 0.6,
		Beta:  0.2,
		Gamma: 0.2,
	}

	resp, err := engine.Recall(ctx, req)
	if err != nil {
		t.Fatalf("recall failed: %v", err)
	}

	if len(resp.Nodes) == 0 {
		t.Fatalf("expected results, got 0")
	}

	rank1 := resp.Nodes[0]
	t.Logf("Rank 1: %s (Score: %f, Sim: %f, Dense: %v, BM25: %v)",
		rank1.ID, rank1.Score, rank1.SimScore, rank1.DenseScore, rank1.BM25Score)

	// Verification 1: Target node ranks #1
	if rank1.ID != "target-consensus-heartbeat" {
		t.Fatalf("expected target-consensus-heartbeat at Rank 1, got %s", rank1.ID)
	}

	// Verification 2: Dense cosine similarity > 0.70
	if rank1.DenseScore == nil || *rank1.DenseScore <= 0.70 {
		t.Fatalf("expected S_dense > 0.70 for conceptual paraphrase, got %v", rank1.DenseScore)
	}
}

func TestRecallEngine_AcceptanceCriterion2_LexicalKeywordFidelity(t *testing.T) {
	ctx := context.Background()
	ms := newMockStore()
	now := time.Now().UTC()

	// Target 1: contains exact technical token "worker_connections 4096"
	targetNginx := model.Node{
		ID:             "nginx-worker-connections",
		EntityType:     "procedure",
		Label:          "Web Gateway High Concurrency Tuning",
		Summary:        "Set worker_connections 4096 in nginx core events block for web tier",
		CreatedAt:      now.Add(-72 * time.Hour),
		LastAccessedAt: now.Add(-72 * time.Hour),
		AccessCount:    3,
		StabilityScore: 1.0,
	}

	// Target 2: contains exact technical token "port 8084"
	targetPort := model.Node{
		ID:             "service-port-8084",
		EntityType:     "decision",
		Label:          "Service Port Bindings",
		Summary:        "Sekha knowledge graph API listening on port 8084",
		CreatedAt:      now.Add(-48 * time.Hour),
		LastAccessedAt: now.Add(-48 * time.Hour),
		AccessCount:    2,
		StabilityScore: 1.0,
	}

	// Frequent distractor nodes
	distractor := model.Node{
		ID:             "distractor-common",
		EntityType:     "concept",
		Label:          "General High Concurrency Overview",
		Summary:        "Overview of high concurrency architecture and web tier gateway load",
		CreatedAt:      now.Add(-2 * time.Minute),
		LastAccessedAt: now.Add(-2 * time.Minute),
		AccessCount:    100,
		StabilityScore: 1.0,
	}

	_, _ = ms.InsertNodes(ctx, []model.Node{targetNginx, targetPort, distractor})

	engine, err := NewEngine(ctx, ms, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	// Query 1: Exact technical tokens "worker_connections 4096" phrased differently
	req1 := model.RecallRequest{
		Query: "tuning parameters for worker_connections 4096",
		TopK:  3,
	}
	resp1, err := engine.Recall(ctx, req1)
	if err != nil {
		t.Fatalf("recall query 1 failed: %v", err)
	}

	if len(resp1.Nodes) == 0 {
		t.Fatalf("query 1 returned 0 nodes")
	}
	if resp1.Nodes[0].ID != "nginx-worker-connections" {
		t.Fatalf("expected nginx-worker-connections at rank 1 via BM25 keyword boost, got %s", resp1.Nodes[0].ID)
	}
	if resp1.Nodes[0].BM25Score == nil || *resp1.Nodes[0].BM25Score <= 0.3 {
		t.Fatalf("expected high BM25 boost for exact technical tokens, got %v", resp1.Nodes[0].BM25Score)
	}

	// Query 2: Technical token "port 8084"
	req2 := model.RecallRequest{
		Query: "knowledge store port 8084",
		TopK:  3,
	}
	resp2, err := engine.Recall(ctx, req2)
	if err != nil {
		t.Fatalf("recall query 2 failed: %v", err)
	}
	if len(resp2.Nodes) == 0 || resp2.Nodes[0].ID != "service-port-8084" {
		t.Fatalf("expected service-port-8084 at rank 1, got %v", resp2.Nodes)
	}
}

func TestRecallEngine_AcceptanceCriterion3_NoiseSuppression(t *testing.T) {
	ctx := context.Background()
	ms := newMockStore()
	now := time.Now().UTC()

	unrelatedNodes := []model.Node{
		{
			ID:             "node-nginx",
			EntityType:     "concept",
			Label:          "Nginx Worker Settings",
			Summary:        "worker_connections 4096 tuned for high concurrent throughput",
			CreatedAt:      now,
			LastAccessedAt: now,
			AccessCount:    10,
			StabilityScore: 1.0,
		},
		{
			ID:             "node-thermal",
			EntityType:     "telemetry",
			Label:          "Cryogenic Refrigerator Telemetry",
			Summary:        "temperature sensor threshold alert on node-3",
			CreatedAt:      now,
			LastAccessedAt: now,
			AccessCount:    5,
			StabilityScore: 1.0,
		},
	}

	_, _ = ms.InsertNodes(ctx, unrelatedNodes)

	engine, err := NewEngine(ctx, ms, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	// Synthetic out-of-vocabulary project query: "Petrelwick deployment"
	req := model.RecallRequest{
		Query: "Petrelwick deployment",
		TopK:  2,
	}

	resp, err := engine.Recall(ctx, req)
	if err != nil {
		t.Fatalf("recall failed: %v", err)
	}

	for _, n := range resp.Nodes {
		t.Logf("Node %s sim_score = %f (Dense: %v, BM25: %v)", n.ID, n.SimScore, n.DenseScore, n.BM25Score)
		// Acceptance Criterion 3: similarity score < 0.10
		if n.SimScore >= 0.10 {
			t.Fatalf("Acceptance Criterion 3 Failed: expected sim_score < 0.10, got %f for %s", n.SimScore, n.ID)
		}
		if n.DenseScore != nil && *n.DenseScore >= 0.10 {
			t.Fatalf("expected dense similarity < 0.10, got %f for %s", *n.DenseScore, n.ID)
		}
	}
}

func TestRecallEngine_HybridModesAndAlpha(t *testing.T) {
	ctx := context.Background()
	ms := newMockStore()
	now := time.Now().UTC()

	node := model.Node{
		ID:             "hybrid-test-node",
		EntityType:     "concept",
		Label:          "Cluster Consensus Heartbeat",
		Summary:        "consensus heartbeat timeout configured to 500ms for node health",
		CreatedAt:      now,
		LastAccessedAt: now,
		AccessCount:    1,
		StabilityScore: 1.0,
	}

	_, _ = ms.InsertNodes(ctx, []model.Node{node})

	engine, err := NewEngine(ctx, ms, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	query := "cluster coordination interval"

	// 1. Pure Dense Mode
	respDense, err := engine.Recall(ctx, model.RecallRequest{
		Query: query,
		Mode:  "dense",
	})
	if err != nil {
		t.Fatalf("recall dense failed: %v", err)
	}
	if math.Abs(respDense.Nodes[0].SimScore-*respDense.Nodes[0].DenseScore) > 1e-4 {
		t.Fatalf("in dense mode, sim_score must equal dense_score")
	}

	// 2. Pure BM25 Mode
	respBM25, err := engine.Recall(ctx, model.RecallRequest{
		Query: query,
		Mode:  "bm25",
	})
	if err != nil {
		t.Fatalf("recall bm25 failed: %v", err)
	}
	if math.Abs(respBM25.Nodes[0].SimScore-*respBM25.Nodes[0].BM25Score) > 1e-4 {
		t.Fatalf("in bm25 mode, sim_score must equal bm25_score")
	}

	// 3. Custom Hybrid Alpha = 0.80
	alpha80 := 0.80
	respCustom, err := engine.Recall(ctx, model.RecallRequest{
		Query:       query,
		HybridAlpha: &alpha80,
	})
	if err != nil {
		t.Fatalf("recall custom failed: %v", err)
	}
	expectedSim := 0.80*(*respCustom.Nodes[0].DenseScore) + 0.20*(*respCustom.Nodes[0].BM25Score)
	if math.Abs(respCustom.Nodes[0].SimScore-math.Round(expectedSim*10000)/10000) > 1e-3 {
		t.Fatalf("expected sim_score %.4f, got %.4f", expectedSim, respCustom.Nodes[0].SimScore)
	}
}

