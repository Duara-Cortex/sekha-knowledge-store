package store

import (
	"context"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
)

// NodeHeader provides lightweight metadata and embedding for in-memory associative indexing.
type NodeHeader struct {
	ID             string
	EntityType     string
	Embedding      []float32
	Magnitude      float32
	LastAccessedAt time.Time
	AccessCount    int64
	StabilityScore float64
}

// Store defines persistence operations for the relational knowledge graph.
type Store interface {
	// Schema & Lifecycle
	Close() error

	// Ingestion / Persistence
	InsertNodes(ctx context.Context, nodes []model.Node) (int, error)
	InsertEdges(ctx context.Context, edges []model.Edge) (int, error)

	// Retrieval
	GetNode(ctx context.Context, id string) (*model.Node, error)
	GetNodes(ctx context.Context, ids []string) (map[string]model.Node, error)
	GetAllNodeHeaders(ctx context.Context) ([]NodeHeader, error)
	GetEdgesForNodes(ctx context.Context, nodeIDs []string) ([]model.Edge, error)

	// Usage tracking
	RecordAccess(ctx context.Context, nodeIDs []string, accessTime time.Time) error

	// Telemetry & Metrics
	GetGraphSummary(ctx context.Context) (*model.GraphSummary, error)
	GetCounts(ctx context.Context) (int64, int64, error)
}
