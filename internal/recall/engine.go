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

// EngineConfig configures hyperparameter defaults, decay constants, and embedding clients.
type EngineConfig struct {
	DefaultAlpha        float64            // Semantic similarity weight w_sim (default: 0.6)
	DefaultBeta         float64            // Usage frequency weight w_freq (default: 0.2)
	DefaultGamma        float64            // Recency decay weight w_rec (default: 0.2)
	DefaultAnchorWeight float64            // Weight for anchor score bonus w_anc (default: 1.0)
	DefaultHybridAlpha  float64            // Hybrid dense semantic vs BM25 weight (default: 0.65)
	DecayHalfLife       time.Duration      // Time constant tau for recency decay (default: 24h)
	FrequencyMaxCount   float64            // Access count normalisation baseline (default: 100.0)
	NeighbourHopBoost   float64            // Attenuation factor for 1-hop expansion (default: 0.35)
	MaxCandidatePool    int                // Max top vector seeds for graph expansion (default: 20)
	VectorDims          int                // Dimensionality for dynamic query embeddings (default: 384)
	Embedder            embedding.Embedder // Dense vector embedder client
}

// DefaultConfig provides balanced cognitive recall defaults.
func DefaultConfig() EngineConfig {
	return EngineConfig{
		DefaultAlpha:        0.6,
		DefaultBeta:         0.2,
		DefaultGamma:        0.2,
		DefaultAnchorWeight: 1.0,
		DefaultHybridAlpha:  0.65,
		DecayHalfLife:       24 * time.Hour,
		FrequencyMaxCount:   100.0,
		NeighbourHopBoost:   0.35,
		MaxCandidatePool:    20,
		VectorDims:          model.DefaultVectorDim, // 384
		Embedder:            nil,
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
	docLen         float64
	embedding      []float32 // Pre-normalised to unit Euclidean length
	magnitude      float32   // Always 1.0 for valid pre-normalised vectors
	lastAccessedAt time.Time
	accessCount    int64
	stabilityScore float64
	anchors        map[string]struct{}
}

func buildTokenSets(label, summary string) (string, string, map[string]bool, map[string]bool, float64) {
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

	// Effective doc length for BM25: label terms carry 2x weight
	docLen := float64(len(labelTokens))*2.0 + float64(len(summaryTokens))
	if docLen < 1.0 {
		docLen = 1.0
	}

	return lowerLabel, lowerSummary, labelSet, summarySet, docLen
}

// bm25Posting records a node's term frequency for a specific token.
type bm25Posting struct {
	nodeID string
	tf     float64
}

// bm25Index holds term statistics and inverted lists for in-memory lexical BM25 ranking.
type bm25Index struct {
	totalDocs   int
	totalDocLen float64
	avgDocLen   float64
	docFreq     map[string]int           // term -> number of documents containing term
	postings    map[string][]bm25Posting // term -> postings list
}

func newBM25Index(nodes map[string]*cachedNode) *bm25Index {
	idx := &bm25Index{
		totalDocs: len(nodes),
		docFreq:   make(map[string]int),
		postings:  make(map[string][]bm25Posting),
	}

	if idx.totalDocs == 0 {
		idx.avgDocLen = 1.0
		return idx
	}

	var sumDocLen float64
	for _, cn := range nodes {
		sumDocLen += cn.docLen

		// Accumulate term frequencies within this document
		tfMap := make(map[string]float64)
		for t := range cn.labelSet {
			tfMap[t] += 2.0 // Label match weight
		}
		for t := range cn.summarySet {
			tfMap[t] += 1.0 // Summary match weight
		}

		for t, tf := range tfMap {
			idx.docFreq[t]++
			idx.postings[t] = append(idx.postings[t], bm25Posting{
				nodeID: cn.id,
				tf:     tf,
			})
		}
	}

	idx.totalDocLen = sumDocLen
	idx.avgDocLen = sumDocLen / float64(idx.totalDocs)
	if idx.avgDocLen <= 0 {
		idx.avgDocLen = 1.0
	}

	return idx
}

// score calculates normalised BM25 scores in [0.0, 1.0] for query tokens across all matching nodes.
func (idx *bm25Index) score(queryTokens []string, queryJoined string, nodes map[string]*cachedNode) map[string]float64 {
	scores := make(map[string]float64)
	if idx == nil || idx.totalDocs == 0 || len(queryTokens) == 0 {
		return scores
	}

	const (
		k1 = 1.2
		b  = 0.75
	)

	N := float64(idx.totalDocs)
	var maxTheoreticalScore float64

	isTechnical := func(t string) bool {
		return strings.ContainsAny(t, "_-.:") || strings.ContainsAny(t, "0123456789")
	}

	for _, qt := range queryTokens {
		postings, ok := idx.postings[qt]
		if !ok || len(postings) == 0 {
			continue
		}

		n := float64(idx.docFreq[qt])
		// Robertson-Spärck Jones IDF (always positive)
		idf := math.Log((N-n+0.5)/(n+0.5) + 1.0)
		if idf < 0.1 {
			idf = 0.1
		}

		termWeight := 1.0
		if isTechnical(qt) {
			termWeight = 2.5 // Boost technical identifiers and parameters
		}

		maxTheoreticalScore += idf * (k1 + 1.0) * termWeight

		for _, p := range postings {
			cn, exists := nodes[p.nodeID]
			if !exists {
				continue
			}

			// BM25 document length normalization
			K := k1 * (1.0 - b + b*(cn.docLen/idx.avgDocLen))
			tfNorm := (p.tf * (k1 + 1.0)) / (p.tf + K)

			scores[p.nodeID] += idf * tfNorm * termWeight
		}
	}

	if maxTheoreticalScore <= 0 {
		return scores
	}

	// Exact phrase bonus if query has multiple tokens
	if queryJoined != "" && len(queryTokens) > 1 {
		for nodeID, raw := range scores {
			cn, exists := nodes[nodeID]
			if !exists {
				continue
			}
			if strings.Contains(cn.lowerLabel, queryJoined) {
				scores[nodeID] = raw + 0.35*maxTheoreticalScore
			} else if strings.Contains(cn.lowerSummary, queryJoined) {
				scores[nodeID] = raw + 0.15*maxTheoreticalScore
			}
		}
	}

	// Normalise to [0.0, 1.0]
	for nodeID, raw := range scores {
		norm := raw / maxTheoreticalScore
		if norm > 1.0 {
			norm = 1.0
		} else if norm < 0.0 {
			norm = 0.0
		}
		scores[nodeID] = norm
	}

	return scores
}

// dotProduct calculates the inner product between two float32 slices using 8-way loop unrolling.
func dotProduct(a, b []float32) float64 {
	n := len(a)
	if n != len(b) || n == 0 {
		return 0.0
	}

	var sum0, sum1, sum2, sum3 float32
	var sum4, sum5, sum6, sum7 float32

	i := 0
	for ; i <= n-8; i += 8 {
		sum0 += a[i] * b[i]
		sum1 += a[i+1] * b[i+1]
		sum2 += a[i+2] * b[i+2]
		sum3 += a[i+3] * b[i+3]
		sum4 += a[i+4] * b[i+4]
		sum5 += a[i+5] * b[i+5]
		sum6 += a[i+6] * b[i+6]
		sum7 += a[i+7] * b[i+7]
	}

	total := sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7
	for ; i < n; i++ {
		total += a[i] * b[i]
	}

	return float64(total)
}

// normalizeVector pre-normalises a vector in-place to unit Euclidean length.
func normalizeVector(vec []float32) ([]float32, float32) {
	if len(vec) == 0 {
		return nil, 0.0
	}
	var sum float64
	for _, v := range vec {
		sum += float64(v * v)
	}
	mag := float32(math.Sqrt(sum))
	if mag > 0 {
		normVec := make([]float32, len(vec))
		for i, v := range vec {
			normVec[i] = v / mag
		}
		return normVec, 1.0
	}
	return vec, 0.0
}

// Engine implements the associative recall algorithm, in-memory vector index, and BM25 index.
type Engine struct {
	store store.Store
	cfg   EngineConfig
	mu    sync.RWMutex
	nodes map[string]*cachedNode
	index []*cachedNode
	bm25  *bm25Index
}

// NewEngine creates and hydrates the associative recall engine from the underlying store.
func NewEngine(ctx context.Context, s store.Store, cfg EngineConfig) (*Engine, error) {
	if cfg.DefaultAlpha <= 0 && cfg.DefaultBeta <= 0 && cfg.DefaultGamma <= 0 {
		cfg = DefaultConfig()
	}
	if cfg.DefaultHybridAlpha <= 0 {
		cfg.DefaultHybridAlpha = 0.65
	}
	if cfg.VectorDims <= 0 {
		cfg.VectorDims = model.DefaultVectorDim
	}
	if cfg.Embedder == nil {
		client := embedding.NewClientFromEnv()
		if client.IsEnabled() {
			cfg.Embedder = client
		}
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

// SetEmbedder dynamically configures the dense embedding engine.
func (e *Engine) SetEmbedder(emb embedding.Embedder) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.cfg.Embedder = emb
}

// Embedder returns the active embedding engine client.
func (e *Engine) Embedder() embedding.Embedder {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.cfg.Embedder
}

// Hydrate populates the in-memory vector index and BM25 index from SQLite.
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

		lowerLabel, lowerSummary, labelSet, summarySet, docLen := buildTokenSets(h.Label, h.Summary)
		normEmb, mag := normalizeVector(h.Embedding)
		if len(normEmb) == 0 {
			text := strings.TrimSpace(h.Label + " " + h.Summary)
			if text != "" {
				dims := e.cfg.VectorDims
				if dims <= 0 {
					dims = model.DefaultVectorDim
				}
				if e.cfg.Embedder != nil {
					if emb, err := e.cfg.Embedder.EmbedText(ctx, text); err == nil && len(emb) > 0 {
						normEmb, mag = normalizeVector(emb)
					}
				}
				if len(normEmb) == 0 {
					normEmb, mag = normalizeVector(embedding.Generate(text, dims))
				}
			}
		}

		cn := &cachedNode{
			id:             h.ID,
			entityType:     h.EntityType,
			label:          h.Label,
			summary:        h.Summary,
			lowerLabel:     lowerLabel,
			lowerSummary:   lowerSummary,
			labelSet:       labelSet,
			summarySet:     summarySet,
			docLen:         docLen,
			embedding:      normEmb,
			magnitude:      mag,
			lastAccessedAt: h.LastAccessedAt,
			accessCount:    h.AccessCount,
			stabilityScore: h.StabilityScore,
			anchors:        anchorSet,
		}
		e.nodes[cn.id] = cn
		e.index = append(e.index, cn)
	}

	e.bm25 = newBM25Index(e.nodes)
	return nil
}

// RegisterNodes updates the in-memory vector index and rebuilds lexical statistics when new nodes arrive.
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

		normEmb, mag := normalizeVector(n.Embedding)
		if len(normEmb) == 0 {
			text := strings.TrimSpace(n.Label + " " + n.Summary)
			if text != "" {
				dims := e.cfg.VectorDims
				if dims <= 0 {
					dims = model.DefaultVectorDim
				}
				if e.cfg.Embedder != nil {
					if emb, err := e.cfg.Embedder.EmbedText(context.Background(), text); err == nil && len(emb) > 0 {
						normEmb, mag = normalizeVector(emb)
					}
				}
				if len(normEmb) == 0 {
					normEmb, mag = normalizeVector(embedding.Generate(text, dims))
				}
			}
		}
		lowerLabel, lowerSummary, labelSet, summarySet, docLen := buildTokenSets(n.Label, n.Summary)

		cn, exists := e.nodes[n.ID]
		if exists {
			cn.entityType = n.EntityType
			cn.label = n.Label
			cn.summary = n.Summary
			cn.lowerLabel = lowerLabel
			cn.lowerSummary = lowerSummary
			cn.labelSet = labelSet
			cn.summarySet = summarySet
			cn.docLen = docLen
			if len(normEmb) > 0 {
				cn.embedding = normEmb
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
				docLen:         docLen,
				embedding:      normEmb,
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

	e.bm25 = newBM25Index(e.nodes)
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
	denseScore   float64
	bm25Score    float64
	freqScore    float64
	recencyScore float64
	anchorScore  float64
	hopDistance  int
}

// Recall executes the associative retrieval algorithm across dense semantic similarity,
// lexical BM25 matching, usage frequency, recency decay, anchor tags, and 1-hop graph expansion.
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

	// Rehydrate in-memory index if empty
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

	// Extract inline hashtags from req.Query if present
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

	// Dynamically generate dense embedding if omitted
	if len(req.Embedding) == 0 && cleanQueryText != "" {
		e.mu.RLock()
		embClient := e.cfg.Embedder
		e.mu.RUnlock()

		if embClient != nil {
			emb, err := embClient.EmbedText(ctx, cleanQueryText)
			if err == nil && len(emb) > 0 {
				req.Embedding = emb
			}
		}

		if len(req.Embedding) == 0 {
			dims := e.cfg.VectorDims
			if dims <= 0 {
				dims = model.DefaultVectorDim
			}
			e.mu.RLock()
			if len(e.index) > 0 && len(e.index[0].embedding) > 0 {
				dims = len(e.index[0].embedding)
			}
			e.mu.RUnlock()
			req.Embedding = embedding.Generate(cleanQueryText, dims)
		}
	}

	// Pre-normalise query embedding
	if len(req.Embedding) > 0 {
		normQ, _ := normalizeVector(req.Embedding)
		req.Embedding = normQ
	}

	// Determine hybrid alpha and search mode
	hybridAlpha := req.GetHybridAlpha(e.cfg.DefaultHybridAlpha)
	mode := strings.ToLower(strings.TrimSpace(req.Mode))
	switch mode {
	case "dense":
		hybridAlpha = 1.0
	case "bm25":
		hybridAlpha = 0.0
	case "hybrid", "":
		// Keep configured hybridAlpha
	}
	if hybridAlpha < 0.0 {
		hybridAlpha = 0.0
	} else if hybridAlpha > 1.0 {
		hybridAlpha = 1.0
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
		tauSeconds = 86400
	}
	logMaxFreq := math.Log(1.0 + e.cfg.FrequencyMaxCount)

	e.mu.RLock()
	// Precompute BM25 lexical scores across all nodes
	var bm25Scores map[string]float64
	if e.bm25 != nil && len(queryTokens) > 0 {
		bm25Scores = e.bm25.score(queryTokens, queryJoined, e.nodes)
	}

	candidates := make(map[string]*candidateScore, len(e.index))
	var scoredList []*candidateScore

	for _, cn := range e.index {
		var simScore float64
		var vecSim float64
		var bm25Sim float64

		if req.EntityID != "" && cn.id == req.EntityID {
			simScore = 1.0
			vecSim = 1.0
			bm25Sim = 1.0
		} else {
			// 1a. Dense Vector Cosine Similarity (pre-normalised inner product)
			if len(req.Embedding) > 0 && len(cn.embedding) == len(req.Embedding) {
				vecSim = dotProduct(req.Embedding, cn.embedding)
				if vecSim < 0.0 {
					vecSim = 0.0
				} else if vecSim > 1.0 {
					vecSim = 1.0
				}
			}

			// 1b. Lexical BM25 Matching
			if bm25Scores != nil {
				bm25Sim = bm25Scores[cn.id]
			}

			// 1c. Hybrid Formulation: S_hybrid = alpha * S_dense + (1 - alpha) * S_bm25
			simScore = hybridAlpha*vecSim + (1.0-hybridAlpha)*bm25Sim
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
			denseScore:   vecSim,
			bm25Score:    bm25Sim,
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

	// 5. Hybrid Semantic Search + 1-Hop Graph Neighbour Expansion
	var relevantEdges []model.Edge
	if expandHops >= 1 && len(seedIDs) > 0 {
		edges, err := e.store.GetEdgesForNodes(ctx, seedIDs)
		if err != nil {
			return nil, fmt.Errorf("failed fetching graph edges for recall: %w", err)
		}
		relevantEdges = edges

		for _, edge := range edges {
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

	// Omit embedding vector array from response unless explicitly requested
	includeEmb := req.IncludeEmbeddings
	if !includeEmb {
		for _, f := range req.Fields {
			if strings.EqualFold(strings.TrimSpace(f), "embedding") {
				includeEmb = true
				break
			}
		}
	}

	scoredNodes := make([]model.ScoredNode, 0, len(topCandidates))
	for _, tc := range topCandidates {
		fullNode, exists := nodesMap[tc.id]
		if !exists {
			continue
		}
		if !includeEmb {
			fullNode.Embedding = nil
		}
		dScore := math.Round(tc.denseScore*10000) / 10000
		bScore := math.Round(tc.bm25Score*10000) / 10000
		scoredNodes = append(scoredNodes, model.ScoredNode{
			Node:           fullNode,
			Score:          math.Round(tc.totalScore*10000) / 10000,
			SimScore:       math.Round(tc.simScore*10000) / 10000,
			DenseScore:     &dScore,
			BM25Score:      &bScore,
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

	// Update access count and last_accessed_at telemetry asynchronously
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

	e.mu.Lock()
	for _, id := range ids {
		if cn, exists := e.nodes[id]; exists {
			cn.accessCount++
			cn.lastAccessedAt = accessTime
		}
	}
	e.mu.Unlock()

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

// Sync compares the active node count in SQLite with the in-memory cache and rehydrates if needed.
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

// StartBackgroundSync periodically checks SQLite for newly consolidated or modified nodes.
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
