package model

import (
	"time"
)

// Outcome status constants for resolved episodic deliberation traces.
const (
	OutcomeSuccess = "success"
	OutcomeFailure = "failure"
	OutcomeNeutral = "neutral"
)

// SensoryItem captures a salient sensory chunk preserved in episodic context.
type SensoryItem struct {
	ID        string    `json:"id,omitempty"`
	Text      string    `json:"text"`
	Salience  float64   `json:"salience"`
	Source    string    `json:"source,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// TrajectoryStep captures a deliberate reasoning step within an episodic trace.
type TrajectoryStep struct {
	StepIndex   int       `json:"step_index"`
	Thought     string    `json:"thought"`
	Action      string    `json:"action,omitempty"`
	Observation string    `json:"observation,omitempty"`
	Status      string    `json:"status"`
	Timestamp   time.Time `json:"timestamp"`
}

// CandidateAction captures proposed or committed actions during deliberation.
type CandidateAction struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	Payload   map[string]interface{} `json:"payload,omitempty"`
	Committed bool                   `json:"committed"`
	CreatedAt time.Time              `json:"created_at"`
}

// EpisodicTrace represents a completed deliberation trajectory dispatched from Node 2.
type EpisodicTrace struct {
	ID               string            `json:"id"`
	SessionID        string            `json:"session_id"`
	TaskGoal         string            `json:"task_goal"`
	Outcome          string            `json:"outcome"`
	SensoryContext   []SensoryItem     `json:"sensory_context,omitempty"`
	Trajectory       []TrajectoryStep  `json:"trajectory,omitempty"`
	CandidateActions []CandidateAction `json:"candidate_actions,omitempty"`
	Consolidated     bool              `json:"consolidated"`
	CreatedAt        time.Time         `json:"created_at"`
	ConsolidatedAt   *time.Time        `json:"consolidated_at,omitempty"`
}

// ConsolidateRequest defines the ingestion payload for POST /api/v1/memory/consolidate.
type ConsolidateRequest struct {
	TraceID          string            `json:"trace_id,omitempty"`
	SessionID        string            `json:"session_id"`
	TaskGoal         string            `json:"task_goal,omitempty"`
	ActiveGoal       string            `json:"active_goal,omitempty"` // Fallback for Node 2 working memory state
	Outcome          string            `json:"outcome,omitempty"`
	Status           string            `json:"status,omitempty"`      // Fallback for Node 2 status
	SensoryContext   []SensoryItem     `json:"sensory_context,omitempty"`
	Trajectory       []TrajectoryStep  `json:"trajectory,omitempty"`
	CandidateActions []CandidateAction `json:"candidate_actions,omitempty"`
	Synchronous      bool              `json:"synchronous,omitempty"` // If true, triggers immediate inline fusion
}

// ConsolidateResponse returns ingestion receipts and consolidation statistics.
type ConsolidateResponse struct {
	Status            string `json:"status"`
	TraceID           string `json:"trace_id"`
	Message           string `json:"message"`
	Synchronous       bool   `json:"synchronous"`
	EntitiesExtracted int    `json:"entities_extracted,omitempty"`
	NodesFused        int    `json:"nodes_fused,omitempty"`
	EdgesReinforced   int    `json:"edges_reinforced,omitempty"`
	CreatedNodes      []Node `json:"created_nodes,omitempty"`
}

// ExtractedEntity represents a salient concept, decision, or entity derived from an episodic trace.
type ExtractedEntity struct {
	ID         string    `json:"id"`
	EntityType string    `json:"entity_type"`
	Label      string    `json:"label"`
	Summary    string    `json:"summary"`
	Embedding  []float32 `json:"embedding,omitempty"`
	Salience   float64   `json:"salience"`
}

// ExtractedRelation represents a causal, temporal, or co-activation link between extracted entities.
type ExtractedRelation struct {
	SourceID     string  `json:"source_id"`
	TargetID     string  `json:"target_id"`
	RelationType string  `json:"relation_type"`
	Weight       float64 `json:"weight"`
}

// DecayConfig parametrizes the mathematical recency decay and Hebbian reinforcement engine.
type DecayConfig struct {
	DecayHalfLife         time.Duration `json:"decay_half_life"`         // Recency decay half-life duration (tau)
	PruneThreshold        float64       `json:"prune_threshold"`         // Omega_prune for node soft-archival
	EdgePruneThreshold    float64       `json:"edge_prune_threshold"`    // Minimum edge weight before removal
	InactivityGracePeriod time.Duration `json:"inactivity_grace_period"` // Period before unreinforced node is pruned
	HebbianLearningRate   float64       `json:"hebbian_learning_rate"`   // Delta W reinforcement multiplier (eta)
	MaxEdgeWeight         float64       `json:"max_edge_weight"`         // Upper bound for edge weight saturation
	BatchSize             int           `json:"batch_size"`              // Batch size for SQLite updates
}

// DefaultDecayConfig returns standard parameters calibrated for 8GB Raspberry Pi 5 operation.
func DefaultDecayConfig() DecayConfig {
	return DecayConfig{
		DecayHalfLife:         72 * time.Hour,
		PruneThreshold:        0.10,
		EdgePruneThreshold:    0.05,
		InactivityGracePeriod: 168 * time.Hour, // 7 days
		HebbianLearningRate:   0.15,
		MaxEdgeWeight:         5.0,
		BatchSize:             500,
	}
}

// ConsolidationStats provides retention, pruning, and graph growth metrics.
type ConsolidationStats struct {
	TotalCycles         int64     `json:"total_cycles"`
	LastCycleAt         time.Time `json:"last_cycle_at"`
	LastCycleDurationMS float64   `json:"last_cycle_duration_ms"`
	TracesProcessed     int64     `json:"traces_processed"`
	ActiveNodes         int64     `json:"active_nodes"`
	ArchivedNodes       int64     `json:"archived_nodes"`
	ActiveEdges         int64     `json:"active_edges"`
	PrunedEdges         int64     `json:"pruned_edges"`
	MeanStabilityScore  float64   `json:"mean_stability_score"`
	MeanEdgeWeight      float64   `json:"mean_edge_weight"`
	DBSizeBytes         int64     `json:"db_size_bytes"`
}

// ConsolidationTriggerResponse is returned when an out-of-band consolidation cycle is manually triggered.
type ConsolidationTriggerResponse struct {
	Status          string             `json:"status"`
	Message         string             `json:"message"`
	DurationMS      float64            `json:"duration_ms"`
	TracesFused     int                `json:"traces_fused"`
	NodesDecayed    int                `json:"nodes_decayed"`
	NodesArchived   int                `json:"nodes_archived"`
	EdgesReinforced int                `json:"edges_reinforced"`
	EdgesPruned     int                `json:"edges_pruned"`
	Stats           ConsolidationStats `json:"stats"`
}
