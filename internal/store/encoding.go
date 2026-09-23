package store

import (
	"encoding/binary"
	"math"
)

// PackVector serialises a slice of float32 into a compact IEEE-754 little-endian binary blob.
// Supports variable dimensions including dense 384-D (1,536 bytes) and legacy 64-D (256 bytes).
func PackVector(vec []float32) []byte {
	if len(vec) == 0 {
		return nil
	}
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// UnpackVector reconstructs a slice of float32 from an IEEE-754 little-endian binary blob.
// Determines vector dimension dynamically based on byte slice length: dim = len(b) / 4.
// Safely handles legacy 64-D blobs (256 bytes), dense 384-D blobs (1,536 bytes),
// and safely rejects malformed or unaligned byte slices without panicking.
func UnpackVector(buf []byte) []float32 {
	if len(buf) == 0 || len(buf)%4 != 0 {
		return nil
	}
	count := len(buf) / 4
	vec := make([]float32, count)
	for i := 0; i < count; i++ {
		vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return vec
}

// EncodeEmbedding serialises a slice of float32 into a compact binary blob.
// Kept for backwards compatibility; delegates to PackVector.
func EncodeEmbedding(vec []float32) []byte {
	return PackVector(vec)
}

// DecodeEmbedding reconstructs a slice of float32 from a binary blob.
// Kept for backwards compatibility; delegates to UnpackVector.
func DecodeEmbedding(buf []byte) []float32 {
	return UnpackVector(buf)
}
