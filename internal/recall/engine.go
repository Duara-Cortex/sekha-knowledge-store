package recall

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/embedding"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

// EngineConfig configures hyperparameter defaults and decay constants.
type EngineConfig struct {
	DefaultAlpha        float64       // Semantic similarity weight (default: 0.6)
	DefaultBeta         float64       // Usage frequency weight (default: 0.2)
	DefaultGamma        float64       // Recency decay weight (default: 0.2)
	DefaultAnchorWeight float64       // Weight for anchor score bonus (default: 1.0)
	DecayHalfLife       time.Duration // Time constant tau for recency decay (default: 24h)
	FrequencyMaxCount   float64       // Access count normalisation baseline (default: 100.0)
	NeighbourHopBoost   float64       // Attenuation factor for 1-hop expansion (default: 0.35)
	MaxCandidatePool    int           // Max top vector seeds for graph expansion (default: 20)
	VectorDims          int           // Dimensionality for dynamic query embeddings (default: 64)
}

// DefaultConfig provides balanced cognitive recall defaults.
func DefaultConfig() EngineConfig {
	return EngineConfig{
		DefaultAlpha:        0.6,
		DefaultBeta:         0.2,
		DefaultGamma:        0.2,
		DefaultAnchorWeight: 1.0,
		DecayHalfLife:       24 * time.Hour,
		FrequencyMaxCount:   100.0,
		NeighbourHopBoost:   0.35,
		MaxCandidatePool:    20,
		VectorDims:          64,
	}
}

type cachedNode struct {
	id             string
	entityType     string
	label          string
	summary        string
	lowerLabel     string
	lowerSummary   string
	labelSet       map[string]bool
	summarySet     map[string]bool
	embedding      []float32
	magnitude      float32
	lastAccessedAt time.Time
	accessCount    int64
	stabilityScore float64
	anchors        map[string]struct{}
}

func buildTokenSets(label, summary string) (string, string, map[string]bool, map[string]bool) {
	lowerLabel := strings.ToLower(label)
	lowerSummary := strings.ToLower(summary)

	labelTokens := embedding.ExtractTokens(lowerLabel)
	summaryTokens := embedding.ExtractTokens(lowerSummary)

	labelSet := make(map[string]bool, len(labelTokens))
	for _, t := range labelTokens {
		labelSet[t] = true
	}

	summarySet := make(map[string]bool, len(summaryTokens))
	for _, t := range summaryTokens {
		summarySet[t] = true
	}

	return lowerLabel, lowerSummary, labelSet, summarySet
}

// Engine implements the associative recall algorithm and in-memory vector index.
type Engine struct {
	store store.Store
	cfg   EngineConfig
	mu    sync.RWMutex
	nodes map[string]*cachedNode
	index []*cachedNode
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
		if h.IsArchived {
			continue
		}
		anchorSet := make(map[string]struct{}, len(h.Anchors))
		for _, a := range h.Anchors {
			clean := model.NormalizeAnchor(a)
			if clean != "" {
				anchorSet[clean] = struct{}{}
			}
		}
		lowerLabel, lowerSummary, labelSet, summarySet := buildTokenSets(h.Label, h.Summary)
		cn := &cachedNode{
			id:             h.ID,
			entityType:     h.EntityType,
			label:          h.Label,
			summary:        h.Summary,
			lowerLabel:     lowerLabel,
			lowerSummary:   lowerSummary,
			labelSet:       labelSet,
			summarySet:     summarySet,
			embedding:      h.Embedding,
			magnitude:      h.Magnitude,
			lastAccessedAt: h.LastAccessedAt,
			accessCount:    h.AccessCount,
			stabilityScore: h.StabilityScore,
			anchors:        anchorSet,
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
		if n.IsArchived {
			if _, exists := e.nodes[n.ID]; exists {
				delete(e.nodes, n.ID)
				newIndex := make([]*cachedNode, 0, len(e.index))
				for _, item := range e.index {
					if item.id != n.ID {
						newIndex = append(newIndex, item)
					}
				}
				e.index = newIndex
			}
			continue
		}

		anchorSet := make(map[string]struct{}, len(n.Anchors))
		for _, a := range n.Anchors {
			clean := model.NormalizeAnchor(a)
			if clean != "" {
				anchorSet[clean] = struct{}{}
			}
		}

		var mag float32
		if len(n.Embedding) > 0 {
			var sum float64
			for _, v := range n.Embedding {
				sum += float64(v * v)
			}
			mag = float32(math.Sqrt(sum))
		}

		lowerLabel, lowerSummary, labelSet, summarySet := buildTokenSets(n.Label, n.Summary)

		cn, exists := e.nodes[n.ID]
		if exists {
			cn.entityType = n.EntityType
			cn.label = n.Label
			cn.summary = n.Summary
			cn.lowerLabel = lowerLabel
			cn.lowerSummary = lowerSummary
			cn.labelSet = labelSet
			cn.summarySet = summarySet
			if len(n.Embedding) > 0 {
				cn.embedding = n.Embedding
				cn.magnitude = mag
			}
			cn.stabilityScore = n.StabilityScore
			if len(anchorSet) > 0 {
				if cn.anchors == nil {
					cn.anchors = make(map[string]struct{})
				}
				for a := range anchorSet {
					cn.anchors[a] = struct{}{}
				}
			}
		} else {
			cn = &cachedNode{
				id:             n.ID,
				entityType:     n.EntityType,
				label:          n.Label,
				summary:        n.Summary,
				lowerLabel:     lowerLabel,
				lowerSummary:   lowerSummary,
				labelSet:       labelSet,
				summarySet:     summarySet,
				embedding:      n.Embedding,
				magnitude:      mag,
				lastAccessedAt: n.LastAccessedAt,
				accessCount:    n.AccessCount,
				stabilityScore: n.StabilityScore,
				anchors:        anchorSet,
			}
			e.nodes[n.ID] = cn
			e.index = append(e.index, cn)
		}
	}
}

// AttachAnchors dynamically updates anchor tags for an in-memory cached node.
func (e *Engine) AttachAnchors(nodeID string, anchors []string) {
	if nodeID == "" || len(anchors) == 0 {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()

	cn, exists := e.nodes[nodeID]
	if !exists {
		return
	}
	if cn.anchors == nil {
		cn.anchors = make(map[string]struct{})
	}
	for _, a := range anchors {
		clean := model.NormalizeAnchor(a)
		if clean != "" {
			cn.anchors[clean] = struct{}{}
		}
	}
}

// candidateScore tracks intermediate scores during recall ranking.
type candidateScore struct {
	id           string
	totalScore   float64
	baseScore    float64
	simScore     float64
	freqScore    float64
	recencyScore float64
	anchorScore  float64
	hopDistance  int
}

// Recall executes the associative retrieval algorithm across semantic similarity,
// usage frequency, recency decay, anchor tag linking, and 1-hop graph neighbour expansion.
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

	// Rehydrate in-memory index if empty (e.g. SQLite had 0 nodes at startup or nodes added externally)
	e.mu.RLock()
	indexLen := len(e.index)
	e.mu.RUnlock()
	if indexLen == 0 {
		_ = e.Hydrate(ctx)
	}

	queryText := strings.TrimSpace(req.Query)

	// Resolve target query anchors
	var queryAnchors []string
	seenQueryAnchors := make(map[string]struct{})
	for _, a := range req.Anchors {
		clean := model.NormalizeAnchor(a)
		if clean != "" {
			if _, exists := seenQueryAnchors[clean]; !exists {
				seenQueryAnchors[clean] = struct{}{}
				queryAnchors = append(queryAnchors, clean)
			}
		}
	}

	// Also extract inline hashtags from req.Query if present
	if queryText != "" && strings.Contains(queryText, "#") {
		fields := strings.Fields(queryText)
		for _, f := range fields {
			if strings.HasPrefix(f, "#") {
				clean := model.NormalizeAnchor(f)
				if clean != "" {
					if _, exists := seenQueryAnchors[clean]; !exists {
						seenQueryAnchors[clean] = struct{}{}
						queryAnchors = append(queryAnchors, clean)
					}
				}
			}
		}
	}

	cleanQueryText := queryText
	if len(queryAnchors) > 0 && strings.Contains(queryText, "#") {
		var textParts []string
		for _, word := range strings.Fields(queryText) {
			if !strings.HasPrefix(word, "#") {
				textParts = append(textParts, word)
			}
		}
		if len(textParts) > 0 {
			cleanQueryText = strings.Join(textParts, " ")
		}
	}

	var queryTokens []string
	var queryJoined string
	if cleanQueryText != "" {
		queryTokens = embedding.ExtractTokens(cleanQueryText)
		queryJoined = strings.Join(queryTokens, " ")
	}

	// If embedding was not provided but query text was passed, dynamically generate semantic vector
	if len(req.Embedding) == 0 && cleanQueryText != "" {
		dims := e.cfg.VectorDims
		if dims <= 0 {
			dims = 64
		}
		e.mu.RLock()
		if len(e.index) > 0 && len(e.index[0].embedding) > 0 {
			dims = len(e.index[0].embedding)
		}
		e.mu.RUnlock()
		req.Embedding = embedding.Generate(cleanQueryText, dims)
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

	anchorMode := strings.ToLower(strings.TrimSpace(req.AnchorMode))
	if anchorMode != "filter" {
		anchorMode = "boost"
	}

	wAnc := req.AnchorWeight
	if wAnc <= 0 {
		wAnc = e.cfg.DefaultAnchorWeight
		if wAnc <= 0 {
			wAnc = 1.0
		}
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
		// 1. Semantic Similarity: Sim(q, node)
		var simScore float64
		if req.EntityID != "" && cn.id == req.EntityID {
			simScore = 1.0
		} else {
			// 1a. Vector Cosine Similarity
			var vecSim float64
			if queryMag > 0 && cn.magnitude > 0 && len(req.Embedding) == len(cn.embedding) {
				var dot float64
				for i := range req.Embedding {
					dot += float64(req.Embedding[i] * cn.embedding[i])
				}
				cosine := dot / (float64(queryMag) * float64(cn.magnitude))
				if cosine > 1.0 {
					cosine = 1.0
				} else if cosine < 0.0 {
					cosine = 0.0
				}
				vecSim = cosine
			}

			// 1b. Lexical & Keyword Matching against node label and summary
			var lexSim float64
			if len(queryTokens) > 0 {
				lexSim = embedding.ScoreLexicalFast(queryTokens, queryJoined, cn.lowerLabel, cn.lowerSummary, cn.labelSet, cn.summarySet)
			}

			// Composite semantic signal: vector similarity + lexical overlap
			if vecSim > 0 && lexSim > 0 {
				simScore = 0.65*vecSim + 0.35*lexSim
			} else if vecSim > 0 {
				simScore = vecSim
			} else {
				simScore = lexSim
			}
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

		// 4. Anchor Matching & Scoring
		matchesAnchor := false
		if len(queryAnchors) > 0 {
			for _, qa := range queryAnchors {
				if _, ok := cn.anchors[qa]; ok {
					matchesAnchor = true
					break
				}
			}
		}

		// Mode: "filter" - discard non-matching candidates before ranking
		if len(queryAnchors) > 0 && anchorMode == "filter" && !matchesAnchor {
			continue
		}

		anchorScore := 0.0
		if matchesAnchor {
			anchorScore = 1.0
		}

		// Score(v | q, anchors) = w_sim S_sim + w_rec S_rec + w_freq S_freq + w_anc S_anchor
		totalScore := rScore + (wAnc * anchorScore)

		cs := &candidateScore{
			id:           cn.id,
			totalScore:   totalScore,
			baseScore:    rScore,
			simScore:     simScore,
			freqScore:    freqScore,
			recencyScore: recencyScore,
			anchorScore:  anchorScore,
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
			var seedID, neighbourID string
			seedScore := 0.0

			if cs, isSource := candidates[edge.SourceID]; isSource && cs.hopDistance == 0 {
				seedID = edge.SourceID
				neighbourID = edge.TargetID
				seedScore = cs.totalScore
			} else if ct, isTarget := candidates[edge.TargetID]; isTarget && ct.hopDistance == 0 {
				seedID = edge.TargetID
				neighbourID = edge.SourceID
				seedScore = ct.totalScore
			}

			if seedID != "" && neighbourID != "" && seedID != neighbourID {
				nCand, exists := candidates[neighbourID]
				if exists {
					boost := e.cfg.NeighbourHopBoost * math.Min(1.0, edge.Weight) * seedScore
					newScore := nCand.baseScore + boost + (wAnc * nCand.anchorScore)
					// A 1-hop neighbour cannot exceed the seed node that activated it
					if nCand.baseScore < seedScore && newScore >= seedScore {
						newScore = seedScore * 0.95
					}
					if newScore > nCand.totalScore {
						nCand.totalScore = newScore
						if nCand.hopDistance == 0 && nCand.simScore < 0.2 {
							nCand.hopDistance = 1
						}
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

	// Check if embeddings should be retained
	includeEmb := req.IncludeEmbeddings
	if !includeEmb {
		for _, f := range req.Fields {
			if strings.EqualFold(strings.TrimSpace(f), "embedding") {
				includeEmb = true
				break
			}
		}
	}

	// Build scored node results in ranked order
	scoredNodes := make([]model.ScoredNode, 0, len(topCandidates))
	for _, tc := range topCandidates {
		fullNode, exists := nodesMap[tc.id]
		if !exists {
			continue
		}
		if !includeEmb {
			fullNode.Embedding = nil
		}
		scoredNodes = append(scoredNodes, model.ScoredNode{
			Node:           fullNode,
			Score:          math.Round(tc.totalScore*10000) / 10000,
			SimScore:       math.Round(tc.simScore*10000) / 10000,
			FrequencyScore: math.Round(tc.freqScore*10000) / 10000,
			RecencyScore:   math.Round(tc.recencyScore*10000) / 10000,
			AnchorScore:    math.Round(tc.anchorScore*10000) / 10000,
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

	var projectedNodes []map[string]any
	if len(req.Fields) > 0 {
		projectedNodes = make([]map[string]any, len(scoredNodes))
		for i, sn := range scoredNodes {
			projectedNodes[i] = sn.Project(req.Fields)
		}
	}

	return &model.RecallResponse{
		Nodes:          scoredNodes,
		Edges:          subgraphEdges,
		QueryLatencyMS: elapsed,
		ProjectedNodes: projectedNodes,
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

// IndexLen returns the current number of active cached nodes in memory.
func (e *Engine) IndexLen() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.index)
}

// Sync compares the active node count in SQLite with the in-memory cache
// and re-hydrates if external processes or background jobs inserted new nodes.
func (e *Engine) Sync(ctx context.Context) error {
	activeCount, _, err := e.store.GetCounts(ctx)
	if err != nil {
		return err
	}

	e.mu.RLock()
	currentCount := int64(len(e.index))
	e.mu.RUnlock()

	if activeCount != currentCount {
		return e.Hydrate(ctx)
	}
	return nil
}

// StartBackgroundSync periodically checks SQLite for newly consolidated or modified nodes
// and synchronises the in-memory recall index.
func (e *Engine) StartBackgroundSync(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		interval = 15 * time.Second
	}

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				syncCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				_ = e.Sync(syncCtx)
				cancel()
			}
		}
	}()
}
