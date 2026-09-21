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
	Label          string
	Summary        string
	Embedding      []float32
	Magnitude      float32
	LastAccessedAt time.Time
	AccessCount    int64
	StabilityScore float64
	IsArchived     bool
	Anchors        []string
}

// Store defines persistence operations for the relational knowledge graph.
type Store interface {
	// Schema & Lifecycle
	Close() error

	// Ingestion / Persistence
	InsertNodes(ctx context.Context, nodes []model.Node) (int, error)
	InsertEdges(ctx context.Context, edges []model.Edge) (int, error)

	// Anchors
	AttachAnchors(ctx context.Context, nodeID string, anchors []string) error
	GetAnchorsForNode(ctx context.Context, nodeID string) ([]string, error)
	GetNodeIDsForAnchors(ctx context.Context, anchors []string) ([]string, error)

	// Retrieval
	GetNode(ctx context.Context, id string) (*model.Node, error)
	GetNodes(ctx context.Context, ids []string) (map[string]model.Node, error)
	GetAllNodeHeaders(ctx context.Context) ([]NodeHeader, error)
	GetEdgesForNodes(ctx context.Context, nodeIDs []string) ([]model.Edge, error)
	FindMatchingNode(ctx context.Context, label string, entityType string) (*model.Node, error)

	// Usage tracking
	RecordAccess(ctx context.Context, nodeIDs []string, accessTime time.Time) error

	// Episodic Trace Queueing & Consolidation Lifecycle
	QueueTrace(ctx context.Context, trace model.EpisodicTrace) error
	GetPendingTraces(ctx context.Context, limit int) ([]model.EpisodicTrace, error)
	MarkTraceConsolidated(ctx context.Context, traceID string, consolidatedAt time.Time) error

	// Hebbian Reinforcement & Mathematical Decay
	ReinforceEdge(ctx context.Context, sourceID, targetID, relationType string, deltaW float64, maxWeight float64, reinforcedAt time.Time) (float64, error)
	BoostNodeStability(ctx context.Context, nodeID string, deltaStability float64, reinforcedAt time.Time) error
	ApplyDecayAndPrune(ctx context.Context, cfg model.DecayConfig, refTime time.Time) (decayed int, archived int, prunedEdges int, err error)

	// Telemetry & Metrics
	GetGraphSummary(ctx context.Context) (*model.GraphSummary, error)
	GetCounts(ctx context.Context) (int64, int64, error)
	GetConsolidationStats(ctx context.Context) (*model.ConsolidationStats, error)
}
