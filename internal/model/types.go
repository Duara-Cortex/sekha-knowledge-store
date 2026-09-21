package model

import (
	"strings"
	"time"
)

// Node represents a vertex in the relational knowledge graph.
type Node struct {
	ID               string     `json:"id"`
	EntityType       string     `json:"entity_type"`
	Label            string     `json:"label"`
	Summary          string     `json:"summary"`
	Embedding        []float32  `json:"embedding,omitempty"`
	Anchors          []string   `json:"anchors,omitempty"`
	CreatedAt        time.Time  `json:"created_at"`
	LastAccessedAt   time.Time  `json:"last_accessed_at"`
	LastReinforcedAt time.Time  `json:"last_reinforced_at,omitempty"`
	ArchivedAt       *time.Time `json:"archived_at,omitempty"`
	AccessCount      int64      `json:"access_count"`
	StabilityScore   float64    `json:"stability_score"`
	IsArchived       bool       `json:"is_archived"`
}

// NormalizeAnchor ensures anchor tags are lowercase, trimmed, and prefixed with '#'
// (e.g. "project:kestrel" -> "#project:kestrel", " #PROJECT:KESTREL " -> "#project:kestrel").
func NormalizeAnchor(anchor string) string {
	a := strings.ToLower(strings.TrimSpace(anchor))
	if a == "" || a == "#" {
		return ""
	}
	if !strings.HasPrefix(a, "#") {
		a = "#" + a
	}
	return a
}

// Edge represents a directed, weighted relationship between two nodes.
type Edge struct {
	SourceID         string    `json:"source_id"`
	TargetID         string    `json:"target_id"`
	RelationType     string    `json:"relation_type"`
	Weight           float64   `json:"weight"`
	CreatedAt        time.Time `json:"created_at"`
	LastReinforcedAt time.Time `json:"last_reinforced_at,omitempty"`
}

// ScoredNode wraps a Node with the associative recall score breakdown.
type ScoredNode struct {
	Node
	Score          float64 `json:"score"`
	SimScore       float64 `json:"sim_score"`
	FrequencyScore float64 `json:"frequency_score"`
	RecencyScore   float64 `json:"recency_score"`
	AnchorScore    float64 `json:"anchor_score,omitempty"`
	HopDistance    int     `json:"hop_distance"`
}

// InsertRequest defines the payload for POST /api/v1/memory/insert.
type InsertRequest struct {
	Nodes []Node `json:"nodes"`
	Edges []Edge `json:"edges"`
}

// InsertResponse returns insertion counts and status.
type InsertResponse struct {
	Status        string `json:"status"`
	InsertedNodes int    `json:"inserted_nodes"`
	InsertedEdges int    `json:"inserted_edges"`
	Error         string `json:"error,omitempty"`
}

// RecallRequest defines the payload for POST /api/v1/memory/recall.
type RecallRequest struct {
	Query        string    `json:"query,omitempty"`
	Embedding    []float32 `json:"embedding,omitempty"`
	EntityID     string    `json:"entity_id,omitempty"`
	TopK         int       `json:"top_k,omitempty"`
	Alpha        float64   `json:"alpha,omitempty"`         // Weight for semantic similarity (default 0.6)
	Beta         float64   `json:"beta,omitempty"`          // Weight for access count frequency (default 0.2)
	Gamma        float64   `json:"gamma,omitempty"`         // Weight for recency decay (default 0.2)
	ExpandHops   int       `json:"expand_hops,omitempty"`   // Graph expansion depth: 0 or 1 (default 1)
	Anchors      []string  `json:"anchors,omitempty"`       // Target anchor tags (e.g. ["#project:kestrel"])
	AnchorMode   string    `json:"anchor_mode,omitempty"`   // "boost" | "filter" (default: "boost")
	AnchorWeight float64   `json:"anchor_weight,omitempty"` // Weight for anchor bonus w_anc (default 1.0)
}

// RecallResponse returns ranked contextual nodes and their relational subgraph.
type RecallResponse struct {
	Nodes          []ScoredNode `json:"nodes"`
	Edges          []Edge       `json:"edges"`
	QueryLatencyMS float64      `json:"query_latency_ms"`
}

// GraphSummary provides topological and density metrics for GET /api/v1/memory/graph.
type GraphSummary struct {
	NodeCount    int64            `json:"node_count"`
	EdgeCount    int64            `json:"edge_count"`
	EntityTypes  map[string]int64 `json:"entity_types"`
	AvgDegree    float64          `json:"avg_degree"`
	GraphDensity float64          `json:"graph_density"`
	DBSizeBytes  int64            `json:"db_size_bytes"`
}

// HealthResponse reports service status and cluster node metadata.
type HealthResponse struct {
	Status        string `json:"status"`
	Node          string `json:"node"`
	Port          int    `json:"port"`
	Service       string `json:"service"`
	UptimeSeconds int64  `json:"uptime_seconds"`
	NodeCount     int64  `json:"node_count"`
	EdgeCount     int64  `json:"edge_count"`
}
