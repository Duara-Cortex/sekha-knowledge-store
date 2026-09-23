package embedding

import (
	"context"
	"hash/fnv"
	"math"
	"strings"
	"sync"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
)

// MockEmbedder provides deterministic, concept-aware dense semantic embeddings
// for unit testing, CI pipelines, and fallback operation when the edge-local
// embedding service is unavailable.
type MockEmbedder struct {
	dim              int
	mu               sync.RWMutex
	customEmbeddings map[string][]float32
	synonymClusters  map[string]string // phrase/term -> canonical concept
}

// NewMockEmbedder initialises a MockEmbedder with 384-D vector support and built-in technical concepts.
func NewMockEmbedder(dim int) *MockEmbedder {
	if dim <= 0 {
		dim = model.DefaultVectorDim // 384
	}

	m := &MockEmbedder{
		dim:              dim,
		customEmbeddings: make(map[string][]float32),
		synonymClusters:  make(map[string]string),
	}

	m.initDefaultSynonyms()
	return m
}

func (m *MockEmbedder) initDefaultSynonyms() {
	// Cluster coordination & consensus equivalence
	m.RegisterSynonyms("concept:consensus_coordination",
		"cluster coordination", "cluster coordinate", "coordination",
		"consensus heartbeat", "heartbeat timeout", "consensus timeout",
		"raft consensus", "cluster heartbeat", "leader election",
		"heartbeat interval", "election timeout", "coordination interval",
	)

	// Time interval / timeout equivalence
	m.RegisterSynonyms("concept:time_interval",
		"interval", "timeout", "period", "duration", "heartbeat period",
	)

	// Episodic memory & deliberation
	m.RegisterSynonyms("concept:episodic_memory",
		"episodic trace", "episodic memory", "memory trace", "memory traces",
		"deliberation trace", "cognitive recollection", "retrieval trace",
	)

	// Networking & proxy
	m.RegisterSynonyms("concept:network_proxy",
		"reverse proxy", "nginx", "worker_connections", "ingress gateway",
		"socket buffer", "listen port", "proxy buffer",
	)

	// Telemetry & sensors
	m.RegisterSynonyms("concept:telemetry_sensor",
		"thermal sensor", "temperature alarm", "cryogenic refrigerator",
		"sensor telemetry", "optical sensor", "thermal threshold",
	)
}

// RegisterSynonyms registers multiple phrases/terms as belonging to a shared canonical concept.
func (m *MockEmbedder) RegisterSynonyms(canonicalConcept string, terms ...string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, term := range terms {
		clean := strings.ToLower(strings.TrimSpace(term))
		if clean != "" {
			m.synonymClusters[clean] = canonicalConcept
		}
	}
}

// RegisterEmbedding explicitly sets a deterministic vector for a specific text string.
func (m *MockEmbedder) RegisterEmbedding(text string, vec []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()
	clean := strings.ToLower(strings.TrimSpace(text))
	copied := make([]float32, len(vec))
	copy(copied, vec)
	m.customEmbeddings[clean] = copied
}

// Dimension returns the configured embedding dimensionality.
func (m *MockEmbedder) Dimension() int {
	return m.dim
}

// EmbedText generates a deterministic 384-D semantic vector for a single text.
func (m *MockEmbedder) EmbedText(ctx context.Context, text string) ([]float32, error) {
	clean := strings.ToLower(strings.TrimSpace(text))

	m.mu.RLock()
	if custom, ok := m.customEmbeddings[clean]; ok {
		res := make([]float32, len(custom))
		copy(res, custom)
		m.mu.RUnlock()
		return res, nil
	}
	m.mu.RUnlock()

	return m.generate(clean), nil
}

// EmbedBatch generates deterministic 384-D semantic vectors for a batch of texts.
func (m *MockEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	res := make([][]float32, len(texts))
	for i, t := range texts {
		vec, err := m.EmbedText(ctx, t)
		if err != nil {
			return nil, err
		}
		res[i] = vec
	}
	return res, nil
}

// generate produces a unit-normalised float32 vector using canonical concepts and pseudo-orthogonal projections.
func (m *MockEmbedder) generate(text string) []float32 {
	dims := m.dim
	if dims <= 0 {
		dims = model.DefaultVectorDim
	}

	vec := make([]float32, dims)
	if text == "" {
		return vec
	}

	m.mu.RLock()
	// 1. Detect and project canonical concepts (high weight for semantic generalization)
	matchedConcepts := make(map[string]bool)
	for phrase, concept := range m.synonymClusters {
		if strings.Contains(text, phrase) {
			if !matchedConcepts[concept] {
				matchedConcepts[concept] = true
				m.projectDeterministic("concept:"+concept, dims, 8.0, vec)
			}
		}
	}
	m.mu.RUnlock()

	// 2. Project individual word tokens, stems, and subwords
	tokens := ExtractTokens(text)
	for i, token := range tokens {
		stem := extractStem(token)
		wordWeight := float32(1.0 + 0.15*math.Min(5.0, float64(len(token))))
		m.projectDeterministic("s:"+stem, dims, wordWeight*1.2, vec)
		m.projectDeterministic("w:"+token, dims, wordWeight*0.7, vec)

		// Subword character 4-grams for compound words (len >= 5) to suppress short n-gram noise
		runes := []rune(token)
		if len(runes) >= 5 {
			for j := 0; j <= len(runes)-4; j++ {
				ngram := string(runes[j : j+4])
				m.projectDeterministic("ng4:"+ngram, dims, 0.25, vec)
			}
		}

		// Bigrams for local phrase structure
		if i+1 < len(tokens) {
			m.projectDeterministic("bi:"+token+"_"+tokens[i+1], dims, 0.5, vec)
		}
	}

	// 3. Fallback if no distinctive tokens or concepts were found
	if len(matchedConcepts) == 0 && len(tokens) == 0 {
		m.projectDeterministic("raw:"+text, dims, 1.0, vec)
	}

	// 4. Normalise to unit Euclidean length
	var sum float64
	for _, v := range vec {
		sum += float64(v * v)
	}
	mag := float32(math.Sqrt(sum))
	if mag > 0 {
		for i := range vec {
			vec[i] /= mag
		}
	}

	return vec
}

// projectDeterministic maps a feature string deterministically into a D-dimensional pseudo-orthogonal vector.
func (m *MockEmbedder) projectDeterministic(feature string, dims int, weight float32, vec []float32) {
	h := fnv.New64a()
	_, _ = h.Write([]byte(feature))
	state := h.Sum64()
	if state == 0 {
		state = 0x517cc1b727220a95
	}

	for i := 0; i < dims; i++ {
		// xorshift64star pseudo-random number generator
		state ^= state >> 12
		state ^= state << 25
		state ^= state >> 27
		val := state * 0x2545F4914F6CDD1D

		// Normalised pseudo-random float in [-1.0, 1.0]
		f := float32(int32(val>>32)) / 2147483648.0
		vec[i] += f * weight
	}
}
