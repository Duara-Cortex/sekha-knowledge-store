package store

import (
	"math"
	"math/rand"
	"testing"
)

func TestPackUnpackVector_384D(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	vec384 := make([]float32, 384)
	for i := 0; i < 384; i++ {
		vec384[i] = rng.Float32()*2.0 - 1.0
	}

	blob := PackVector(vec384)
	expectedBytes := 384 * 4 // 1,536 bytes
	if len(blob) != expectedBytes {
		t.Fatalf("expected 384-D blob to be %d bytes, got %d", expectedBytes, len(blob))
	}

	unpacked := UnpackVector(blob)
	if len(unpacked) != 384 {
		t.Fatalf("expected unpacked length 384, got %d", len(unpacked))
	}

	for i := 0; i < 384; i++ {
		if math.Abs(float64(unpacked[i]-vec384[i])) > 1e-6 {
			t.Fatalf("mismatch at index %d: expected %f, got %f", i, vec384[i], unpacked[i])
		}
	}
}

func TestPackUnpackVector_Legacy64D_Compatibility(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	vec64 := make([]float32, 64)
	for i := 0; i < 64; i++ {
		vec64[i] = rng.Float32()*2.0 - 1.0
	}

	blob := PackVector(vec64)
	expectedBytes := 64 * 4 // 256 bytes
	if len(blob) != expectedBytes {
		t.Fatalf("expected 64-D blob to be %d bytes, got %d", expectedBytes, len(blob))
	}

	unpacked := UnpackVector(blob)
	if len(unpacked) != 64 {
		t.Fatalf("expected unpacked length 64, got %d", len(unpacked))
	}

	for i := 0; i < 64; i++ {
		if math.Abs(float64(unpacked[i]-vec64[i])) > 1e-6 {
			t.Fatalf("mismatch at index %d: expected %f, got %f", i, vec64[i], unpacked[i])
		}
	}
}

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
	if UnpackVector([]byte{1, 2, 3, 4, 5}) != nil {
		t.Fatalf("expected nil for unaligned blob length in UnpackVector")
	}
}
