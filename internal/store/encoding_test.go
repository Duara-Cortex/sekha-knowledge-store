package store

import (
	"math"
	"testing"
)

func TestEncodeDecodeEmbedding(t *testing.T) {
	original := []float32{0.0, 1.234, -5.678, 3.14159, 100.25}
	blob := EncodeEmbedding(original)
	if len(blob) != len(original)*4 {
		t.Fatalf("expected blob length %d, got %d", len(original)*4, len(blob))
	}

	decoded := DecodeEmbedding(blob)
	if len(decoded) != len(original) {
		t.Fatalf("expected decoded length %d, got %d", len(original), len(decoded))
	}

	for i := range original {
		if math.Abs(float64(decoded[i]-original[i])) > 1e-6 {
			t.Fatalf("mismatch at index %d: expected %f, got %f", i, original[i], decoded[i])
		}
	}

	// Test nil and empty
	if EncodeEmbedding(nil) != nil {
		t.Fatalf("expected nil for nil input")
	}
	if DecodeEmbedding(nil) != nil {
		t.Fatalf("expected nil for nil blob")
	}
	if DecodeEmbedding([]byte{1, 2, 3}) != nil {
		t.Fatalf("expected nil for unaligned blob length")
	}
}
