package recall

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

// EngineConfig configures hyperparameter defaults and decay constants.
type EngineConfig struct {
	DefaultAlpha       float64       // Semantic similarity weight (default: 0.6)
	DefaultBeta        float64       // Usage frequency weight (default: 0.2)
	DefaultGamma       float64       // Recency decay weight (default: 0.2)
	DecayHalfLife      time.Duration // Time constant tau for recency decay (default: 24h)
	FrequencyMaxCount  float64       // Access count normalization baseline (default: 100.0)
	NeighborHopBoost   float64       // Attenuation factor for 1-hop expansion (default: 0.5)
	MaxCandidatePool   int           // Max top vector seeds for graph expansion (default: 20)
}

// DefaultConfig provides balanced cognitive recall defaults.
func DefaultConfig() EngineConfig {
	return EngineConfig{
		DefaultAlpha:      0.6,
		DefaultBeta:       0.2,
		DefaultGamma:      0.2,
		DecayHalfLife:     24 * time.Hour,
		FrequencyMaxCount: 100.0,
		NeighborHopBoost:  0.5,
		MaxCandidatePool:  20,
	}
}

type cachedNode struct {
	id             string
	entityType     string
	embedding      []float32
	magnitude      float32
	lastAccessedAt time.Time
	accessCount    int64
	stabilityScore float64
}

// Engine implements the associative recall algorithm and in-memory vector index.
type Engine struct {
	store  store.Store
	cfg    EngineConfig
	mu     sync.RWMutex
	nodes  map[string]*cachedNode
	index  []*cachedNode
}

// NewEngine creates and hydrates the associative recall engine from the underlying store.
func NewEngine(ctx context.Context, s store.Store, cfg EngineConfig) (*Engine, error) {
	if cfg.DefaultAlpha <= 0 && cfg.DefaultBeta <= 0 && cfg.DefaultGamma <= 0 {
		cfg = DefaultConfig()
	}

	e := &Engine{
		store: s,
		cfg:   cfg,
		nodes: make(map[string]*cachedNode),
	}

	if err := e.Hydrate(ctx); err != nil {
		return nil, fmt.Errorf("failed to hydrate in-memory index: %w", err)
	}

	return e, nil
}

// Hydrate populates the in-memory vector index from the database.
func (e *Engine) Hydrate(ctx context.Context) error {
	headers, err := e.store.GetAllNodeHeaders(ctx)
	if err != nil {
		return err
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	e.nodes = make(map[string]*cachedNode, len(headers))
	e.index = make([]*cachedNode, 0, len(headers))

	for _, h := range headers {
		cn := &cachedNode{
			id:             h.ID,
			entityType:     h.EntityType,
			embedding:      h.Embedding,
			magnitude:      h.Magnitude,
			lastAccessedAt: h.LastAccessedAt,
			accessCount:    h.AccessCount,
			stabilityScore: h.StabilityScore,
		}
		e.nodes[cn.id] = cn
		e.index = append(e.index, cn)
	}

	return nil
}

// RegisterNodes updates the in-memory vector index when new nodes are inserted.
func (e *Engine) RegisterNodes(nodes []model.Node) {
	e.mu.Lock()
	defer e.mu.Unlock()

	for _, n := range nodes {
		var mag float32
		if len(n.Embedding) > 0 {
			var sum float64
			for _, v := range n.Embedding {
				sum += float64(v * v)
			}
			mag = float32(math.Sqrt(sum))
		}

		cn, exists := e.nodes[n.ID]
		if exists {
			cn.entityType = n.EntityType
			if len(n.Embedding) > 0 {
				cn.embedding = n.Embedding
				cn.magnitude = mag
			}
			cn.stabilityScore = n.StabilityScore
		} else {
			cn = &cachedNode{
				id:             n.ID,
				entityType:     n.EntityType,
				embedding:      n.Embedding,
				magnitude:      mag,
				lastAccessedAt: n.LastAccessedAt,
				accessCount:    n.AccessCount,
				stabilityScore: n.StabilityScore,
			}
			e.nodes[n.ID] = cn
			e.index = append(e.index, cn)
		}
	}
}

// candidateScore tracks intermediate scores during recall ranking.
type candidateScore struct {
	id             string
	totalScore     float64
	simScore       float64
	freqScore      float64
	recencyScore   float64
	hopDistance    int
}

// Recall executes the associative retrieval algorithm across semantic similarity,
// usage frequency, recency decay, and 1-hop graph neighbour expansion.
func (e *Engine) Recall(ctx context.Context, req model.RecallRequest) (*model.RecallResponse, error) {
	start := time.Now()

	topK := req.TopK
	if topK <= 0 {
		topK = 5
	}
	expandHops := req.ExpandHops
	if expandHops < 0 {
		expandHops = 1
	}

	alpha := req.Alpha
	beta := req.Beta
	gamma := req.Gamma
	if alpha == 0 && beta == 0 && gamma == 0 {
		alpha = e.cfg.DefaultAlpha
		beta = e.cfg.DefaultBeta
		gamma = e.cfg.DefaultGamma
	}
	// Normalise weights
	totalWeight := alpha + beta + gamma
	if totalWeight > 0 {
		alpha /= totalWeight
		beta /= totalWeight
		gamma /= totalWeight
	} else {
		alpha, beta, gamma = 0.6, 0.2, 0.2
	}

	// Compute query embedding magnitude
	var queryMag float32
	if len(req.Embedding) > 0 {
		var sum float64
		for _, v := range req.Embedding {
			sum += float64(v * v)
		}
		queryMag = float32(math.Sqrt(sum))
	}

	now := time.Now().UTC()
	tauSeconds := e.cfg.DecayHalfLife.Seconds()
	if tauSeconds <= 0 {
		tauSeconds = 86400 // 24 hours default
	}
	logMaxFreq := math.Log(1.0 + e.cfg.FrequencyMaxCount)

	e.mu.RLock()
	candidates := make(map[string]*candidateScore, len(e.index))
	var scoredList []*candidateScore

	for _, cn := range e.index {
		// 1. Semantic Vector Similarity: Sim(q, embedding)
		var simScore float64
		if req.EntityID != "" && cn.id == req.EntityID {
			simScore = 1.0
		} else if queryMag > 0 && cn.magnitude > 0 && len(req.Embedding) == len(cn.embedding) {
			var dot float64
			for i := range req.Embedding {
				dot += float64(req.Embedding[i] * cn.embedding[i])
			}
			cosine := dot / (float64(queryMag) * float64(cn.magnitude))
			// Clamp cosine [-1.0, 1.0] to [0.0, 1.0]
			if cosine > 1.0 {
				cosine = 1.0
			} else if cosine < -1.0 {
				cosine = -1.0
			}
			simScore = (cosine + 1.0) / 2.0
		}

		// 2. Usage Frequency: log(1 + access_count)
		freqScore := 0.0
		if cn.accessCount > 0 {
			freqScore = math.Log(1.0+float64(cn.accessCount)) / logMaxFreq
			if freqScore > 1.0 {
				freqScore = 1.0
			}
		}

		// 3. Recency Decay: exp(-delta_t / tau)
		recencyScore := 1.0
		if !cn.lastAccessedAt.IsZero() {
			delta := now.Sub(cn.lastAccessedAt).Seconds()
			if delta > 0 {
				recencyScore = math.Exp(-delta / tauSeconds)
			}
		}

		// Combined Base Score: R(node) = alpha*Sim + beta*Freq + gamma*Recency
		rScore := alpha*simScore + beta*freqScore + gamma*recencyScore

		// Factor in node stability score
		if cn.stabilityScore > 0 && cn.stabilityScore != 1.0 {
			rScore *= (0.8 + 0.2*cn.stabilityScore)
		}

		cs := &candidateScore{
			id:           cn.id,
			totalScore:   rScore,
			simScore:     simScore,
			freqScore:    freqScore,
			recencyScore: recencyScore,
			hopDistance:  0,
		}
		candidates[cn.id] = cs
		scoredList = append(scoredList, cs)
	}
	e.mu.RUnlock()

	// Sort candidate seeds by initial score
	sort.Slice(scoredList, func(i, j int) bool {
		return scoredList[i].totalScore > scoredList[j].totalScore
	})

	// Select seed candidates for 1-hop expansion
	seedPoolSize := e.cfg.MaxCandidatePool
	if seedPoolSize > len(scoredList) {
		seedPoolSize = len(scoredList)
	}
	seedIDs := make([]string, 0, seedPoolSize)
	for i := 0; i < seedPoolSize; i++ {
		seedIDs = append(seedIDs, scoredList[i].id)
	}

	// 4. Hybrid Semantic Search + 1-Hop Graph Neighbour Expansion
	var relevantEdges []model.Edge
	if expandHops >= 1 && len(seedIDs) > 0 {
		edges, err := e.store.GetEdgesForNodes(ctx, seedIDs)
		if err != nil {
			return nil, fmt.Errorf("failed fetching graph edges for recall: %w", err)
		}
		relevantEdges = edges

		for _, edge := range edges {
			// Find which end is the seed and which is the neighbour
			var seedID, neighborID string
			seedScore := 0.0

			if cs, isSource := candidates[edge.SourceID]; isSource && cs.hopDistance == 0 {
				seedID = edge.SourceID
				neighborID = edge.TargetID
				seedScore = cs.totalScore
			} else if ct, isTarget := candidates[edge.TargetID]; isTarget && ct.hopDistance == 0 {
				seedID = edge.TargetID
				neighborID = edge.SourceID
				seedScore = ct.totalScore
			}

			if seedID != "" && neighborID != "" && seedID != neighborID {
				nCand, exists := candidates[neighborID]
				if exists {
					boost := e.cfg.NeighborHopBoost * math.Min(1.0, edge.Weight) * seedScore
					nCand.totalScore += boost
					if nCand.hopDistance == 0 && nCand.simScore < 0.1 {
						nCand.hopDistance = 1
					}
				}
			}
		}
	}

	// Re-sort all candidate nodes by final boosted score
	finalList := make([]*candidateScore, 0, len(candidates))
	for _, c := range candidates {
		finalList = append(finalList, c)
	}
	sort.Slice(finalList, func(i, j int) bool {
		return finalList[i].totalScore > finalList[j].totalScore
	})

	if topK > len(finalList) {
		topK = len(finalList)
	}
	topCandidates := finalList[:topK]

	// Fetch full node contents from SQLite for the top K results
	targetIDs := make([]string, len(topCandidates))
	for i, tc := range topCandidates {
		targetIDs[i] = tc.id
	}

	nodesMap, err := e.store.GetNodes(ctx, targetIDs)
	if err != nil {
		return nil, fmt.Errorf("failed fetching full nodes: %w", err)
	}

	// Build scored node results in ranked order
	scoredNodes := make([]model.ScoredNode, 0, len(topCandidates))
	for _, tc := range topCandidates {
		fullNode, exists := nodesMap[tc.id]
		if !exists {
			continue
		}
		scoredNodes = append(scoredNodes, model.ScoredNode{
			Node:           fullNode,
			Score:          math.Round(tc.totalScore*10000) / 10000,
			SimScore:       math.Round(tc.simScore*10000) / 10000,
			FrequencyScore: math.Round(tc.freqScore*10000) / 10000,
			RecencyScore:   math.Round(tc.recencyScore*10000) / 10000,
			HopDistance:    tc.hopDistance,
		})
	}

	// Filter relevant edges to only those connecting the recalled nodes
	returnedIDSet := make(map[string]struct{}, len(scoredNodes))
	for _, sn := range scoredNodes {
		returnedIDSet[sn.ID] = struct{}{}
	}

	var subgraphEdges []model.Edge
	seenEdges := make(map[string]struct{})
	for _, edge := range relevantEdges {
		_, srcIn := returnedIDSet[edge.SourceID]
		_, tgtIn := returnedIDSet[edge.TargetID]
		if srcIn || tgtIn {
			key := fmt.Sprintf("%s->%s:%s", edge.SourceID, edge.TargetID, edge.RelationType)
			if _, seen := seenEdges[key]; !seen {
				seenEdges[key] = struct{}{}
				subgraphEdges = append(subgraphEdges, edge)
			}
		}
	}

	// Update access count and last_accessed_at telemetry in background/synchronously
	e.recordAccess(ctx, targetIDs, now)

	elapsed := float64(time.Since(start).Microseconds()) / 1000.0 // Latency in ms

	return &model.RecallResponse{
		Nodes:          scoredNodes,
		Edges:          subgraphEdges,
		QueryLatencyMS: elapsed,
	}, nil
}

func (e *Engine) recordAccess(ctx context.Context, ids []string, accessTime time.Time) {
	if len(ids) == 0 {
		return
	}

	// 1. Update in-memory cache instantly
	e.mu.Lock()
	for _, id := range ids {
		if cn, exists := e.nodes[id]; exists {
			cn.accessCount++
			cn.lastAccessedAt = accessTime
		}
	}
	e.mu.Unlock()

	// 2. Persist to SQLite store asynchronously so query latency is completely unaffected
	go func() {
		bgCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = e.store.RecordAccess(bgCtx, ids, accessTime)
	}()
}
