package embedding

import (
	"testing"
)

func TestGenerate_Identity(t *testing.T) {
	text := "Project Kestrel ingest service settings"
	v1 := Generate(text, 64)
	v2 := Generate(text, 64)

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
