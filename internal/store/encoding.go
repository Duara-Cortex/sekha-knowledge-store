package store

import (
	"encoding/binary"
	"math"
)

// EncodeEmbedding serialises a slice of float32 into a compact binary blob.
func EncodeEmbedding(vec []float32) []byte {
	if len(vec) == 0 {
		return nil
	}
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// DecodeEmbedding reconstructs a slice of float32 from a binary blob.
func DecodeEmbedding(buf []byte) []float32 {
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
