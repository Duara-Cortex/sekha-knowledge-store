package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"
)

func TestGenerate_Identity(t *testing.T) {
	text := "Project Kestrel ingest service settings"
	v1 := Generate(text, 384)
	v2 := Generate(text, 384)

	sim := CosineSimilarity(v1, v2)
	if sim < 0.9999 {
		t.Fatalf("expected cosine similarity 1.0 for identical text, got %f", sim)
	}
}

func TestGenerate_ParaphraseAndOverlap(t *testing.T) {
	// Need to ensure paraphrased/overlapping queries have strong semantic cosine similarity
	text1 := "Project Kestrel ingest service settings"
	text2 := "Project Kestrel ingest service"
	text3 := "Ingest service configuration settings for Project Kestrel"
	unrelated := "Quantum cryogenic refrigerator sensor telemetry"

	v1 := Generate(text1, 64)
	v2 := Generate(text2, 64)
	v3 := Generate(text3, 64)
	vUnrelated := Generate(unrelated, 64)

	sim12 := CosineSimilarity(v1, v2)
	sim13 := CosineSimilarity(v1, v3)
	simUnrelated := CosineSimilarity(v1, vUnrelated)

	t.Logf("sim(v1, v2) = %f", sim12)
	t.Logf("sim(v1, v3) = %f", sim13)
	t.Logf("sim(v1, unrelated) = %f", simUnrelated)

	if sim12 < 0.70 {
		t.Fatalf("expected strong similarity between overlapping text, got %f", sim12)
	}
	if sim13 < 0.70 {
		t.Fatalf("expected strong similarity between reordered/paraphrased text, got %f", sim13)
	}
	if simUnrelated > 0.25 {
		t.Fatalf("expected low similarity to unrelated text, got %f", simUnrelated)
	}
}

func TestSemanticParaphrasing_AcceptanceCriterion1(t *testing.T) {
	ctx := context.Background()
	embedder := NewMockEmbedder(384)

	query := "cluster coordination interval"
	storedFact := "consensus heartbeat timeout configured to 500ms"

	qVec, err := embedder.EmbedText(ctx, query)
	if err != nil {
		t.Fatalf("failed embedding query: %v", err)
	}
	sVec, err := embedder.EmbedText(ctx, storedFact)
	if err != nil {
		t.Fatalf("failed embedding stored fact: %v", err)
	}

	sim := CosineSimilarity(qVec, sVec)
	t.Logf("Dense semantic similarity S_dense: %f (Required: > 0.70)", sim)

	if sim <= 0.70 {
		t.Fatalf("Acceptance Criterion 1 Failed: S_dense was %f, required > 0.70", sim)
	}
}

func TestNoiseSuppression_AcceptanceCriterion3(t *testing.T) {
	ctx := context.Background()
	embedder := NewMockEmbedder(384)

	query := "Petrelwick deployment"
	unrelated1 := "nginx worker_connections 4096"
	unrelated2 := "quantum cryogenic cooling compressor"

	qVec, _ := embedder.EmbedText(ctx, query)
	u1Vec, _ := embedder.EmbedText(ctx, unrelated1)
	u2Vec, _ := embedder.EmbedText(ctx, unrelated2)

	sim1 := CosineSimilarity(qVec, u1Vec)
	sim2 := CosineSimilarity(qVec, u2Vec)

	t.Logf("sim(Petrelwick, nginx worker_connections): %f (Required: < 0.10)", sim1)
	t.Logf("sim(Petrelwick, quantum cryogenic): %f (Required: < 0.10)", sim2)

	if sim1 >= 0.10 {
		t.Fatalf("Acceptance Criterion 3 Failed: sim1 was %f, required < 0.10", sim1)
	}
	if sim2 >= 0.10 {
		t.Fatalf("Acceptance Criterion 3 Failed: sim2 was %f, required < 0.10", sim2)
	}
}

func TestGenerate_MorphologicalVariation(t *testing.T) {
	// Single word inflection
	w1 := Generate("consolidate", 64)
	w2 := Generate("consolidating", 64)
	simWord := CosineSimilarity(w1, w2)
	t.Logf("sim(consolidate, consolidating) single word = %f", simWord)
	if simWord < 0.60 {
		t.Fatalf("expected single word inflections to have high similarity, got %f", simWord)
	}

	// Multi-word phrase with inflections
	v1 := Generate("consolidate memory traces", 64)
	v2 := Generate("consolidating memory trace", 64)

	sim := CosineSimilarity(v1, v2)
	t.Logf("sim(consolidate memory traces, consolidating memory trace) = %f", sim)
	if sim < 0.50 {
		t.Fatalf("expected subword n-grams and stems to provide high similarity for phrase inflections, got %f", sim)
	}
}

func TestScoreLexical(t *testing.T) {
	query := "Project Kestrel ingest settings"
	tokens := ExtractTokens(query)

	label := "Project Kestrel Ingest Service"
	summary := "Deliberation settings for ingest service"

	score := ScoreLexical(tokens, label, summary)
	t.Logf("Lexical score for matching label & summary: %f", score)
	if score < 0.70 {
		t.Fatalf("expected high lexical score for matching tokens, got %f", score)
	}

	unrelatedLabel := "Thermal Sensor Threshold"
	unrelatedSummary := "CPU temperature alarms"
	unrelatedScore := ScoreLexical(tokens, unrelatedLabel, unrelatedSummary)
	t.Logf("Lexical score for unrelated: %f", unrelatedScore)
	if unrelatedScore > 0.10 {
		t.Fatalf("expected zero/low lexical score for unrelated entity, got %f", unrelatedScore)
	}
}

func TestClient_HTTPProtocol(t *testing.T) {
	expectedVec := make([]float32, 384)
	expectedVec[0] = 0.5
	expectedVec[383] = -0.5

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		var req openAIEmbeddingRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("failed decoding body: %v", err)
		}
		resp := openAIEmbeddingResponse{
			Object: "list",
			Data: []openAIEmbeddingItem{
				{
					Index:     0,
					Embedding: expectedVec,
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := Config{
		Enabled:   true,
		URL:       server.URL,
		Dimension: 384,
		Timeout:   2 * time.Second,
	}

	client := NewClient(cfg, nil)
	vec, err := client.EmbedText(context.Background(), "test text")
	if err != nil {
		t.Fatalf("EmbedText failed: %v", err)
	}

	if len(vec) != 384 {
		t.Fatalf("expected 384 dimensions, got %d", len(vec))
	}
	if vec[0] != 0.5 || vec[383] != -0.5 {
		t.Fatalf("vector contents mismatch")
	}

	health := client.CheckHealth(context.Background())
	if health.Status != "reachable" {
		t.Fatalf("expected status reachable, got %s", health.Status)
	}
	if health.URL != server.URL {
		t.Fatalf("expected URL %s, got %s", server.URL, health.URL)
	}
}

func TestClient_BatchHTTPProtocol(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openAIEmbeddingRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		inputs, _ := req.Input.([]any)
		var items []openAIEmbeddingItem
		// Return in reverse index to verify client sorting
		for i := len(inputs) - 1; i >= 0; i-- {
			vec := make([]float32, 384)
			vec[0] = float32(i + 1)
			items = append(items, openAIEmbeddingItem{
				Index:     i,
				Embedding: vec,
			})
		}
		_ = json.NewEncoder(w).Encode(openAIEmbeddingResponse{Data: items})
	}))
	defer server.Close()

	cfg := Config{
		Enabled:   true,
		URL:       server.URL,
		Dimension: 384,
		Timeout:   2 * time.Second,
	}

	client := NewClient(cfg, nil)
	batch, err := client.EmbedBatch(context.Background(), []string{"one", "two", "three"})
	if err != nil {
		t.Fatalf("EmbedBatch failed: %v", err)
	}

	if len(batch) != 3 {
		t.Fatalf("expected 3 batch items, got %d", len(batch))
	}
	for i := 0; i < 3; i++ {
		if batch[i][0] != float32(i+1) {
			t.Fatalf("expected item %d to have value %f, got %f", i, float32(i+1), batch[i][0])
		}
	}
}

func TestClient_FallbackOnUnreachableService(t *testing.T) {
	// Point to unreachable localhost port
	cfg := Config{
		Enabled:   true,
		URL:       "http://localhost:59999/v1/embeddings",
		Dimension: 384,
		Timeout:   100 * time.Millisecond,
	}

	fallback := NewMockEmbedder(384)
	client := NewClient(cfg, fallback)

	vec, err := client.EmbedText(context.Background(), "test fallback text")
	if err != nil {
		t.Fatalf("expected fallback to succeed, got error: %v", err)
	}

	if len(vec) != 384 {
		t.Fatalf("expected 384 dimensions from fallback, got %d", len(vec))
	}

	health := client.CheckHealth(context.Background())
	if health.Status != "degraded" {
		t.Fatalf("expected degraded health status when service is unreachable, got %s", health.Status)
	}
}

func TestConfigFromEnv(t *testing.T) {
	os.Setenv("SEKHA_EMBEDDING_ENABLED", "true")
	os.Setenv("SEKHA_EMBEDDING_URL", "http://embedding.internal:8086/v1/embeddings")
	os.Setenv("SEKHA_EMBEDDING_DIM", "384")
	os.Setenv("SEKHA_EMBEDDING_TIMEOUT_MS", "750")
	defer func() {
		os.Unsetenv("SEKHA_EMBEDDING_ENABLED")
		os.Unsetenv("SEKHA_EMBEDDING_URL")
		os.Unsetenv("SEKHA_EMBEDDING_DIM")
		os.Unsetenv("SEKHA_EMBEDDING_TIMEOUT_MS")
	}()

	cfg := ConfigFromEnv()
	if !cfg.Enabled {
		t.Fatalf("expected enabled=true")
	}
	if cfg.URL != "http://embedding.internal:8086/v1/embeddings" {
		t.Fatalf("unexpected URL: %s", cfg.URL)
	}
	if cfg.Dimension != 384 {
		t.Fatalf("expected dimension 384, got %d", cfg.Dimension)
	}
	if cfg.Timeout != 750*time.Millisecond {
		t.Fatalf("expected timeout 750ms, got %v", cfg.Timeout)
	}
}

func TestConfigFromEnv_DefaultDisabled(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.Enabled {
		t.Fatalf("expected default config to have Enabled=false")
	}
	if cfg.URL != "" {
		t.Fatalf("expected default config to have empty URL, got %s", cfg.URL)
	}

	client := NewClient(cfg, nil)
	health := client.CheckHealth(context.Background())
	if health.Enabled {
		t.Fatalf("expected client health Enabled=false")
	}
	if health.Status != "offline" {
		t.Fatalf("expected client health Status=offline, got %s", health.Status)
	}
}
