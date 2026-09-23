package model

import (
	"encoding/json"
	"strings"
	"time"
)

// DefaultVectorDim specifies standard dense semantic vector dimensionality (all-MiniLM-L6-v2 384-D).
const DefaultVectorDim = 384

// LegacyVectorDim specifies legacy prototype SHA-256 token hash projection dimensionality.
const LegacyVectorDim = 64

// Node represents a vertex in the relational knowledge graph.
type Node struct {
	ID               string     `json:"id"`
	EntityType       string     `json:"entity_type,omitempty"`
	Label            string     `json:"label"`
	Summary          string     `json:"summary,omitempty"`
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
	Score          float64  `json:"score"`
	SimScore       float64  `json:"sim_score"`
	DenseScore     *float64 `json:"dense_score,omitempty"`
	BM25Score      *float64 `json:"bm25_score,omitempty"`
	FrequencyScore float64  `json:"frequency_score"`
	RecencyScore   float64  `json:"recency_score"`
	AnchorScore    float64  `json:"anchor_score,omitempty"`
	HopDistance    int      `json:"hop_distance"`
}

// MarshalJSON customises JSON serialisation for ScoredNode using ScoredNodeDTO to omit nil embeddings and prune schemas.
func (sn ScoredNode) MarshalJSON() ([]byte, error) {
	return json.Marshal(sn.ToDTO(len(sn.Embedding) > 0))
}

// UnmarshalJSON implements custom JSON deserialisation for ScoredNode.
func (sn *ScoredNode) UnmarshalJSON(data []byte) error {
	type Alias ScoredNode
	aux := struct {
		*Alias
	}{
		Alias: (*Alias)(sn),
	}
	return json.Unmarshal(data, &aux)
}

// ScoredNodeDTO is a serialisable representation of ScoredNode supporting projection and embedding omission.
type ScoredNodeDTO struct {
	ID               string     `json:"id,omitempty"`
	EntityType       string     `json:"entity_type,omitempty"`
	Label            string     `json:"label,omitempty"`
	Summary          string     `json:"summary,omitempty"`
	Embedding        []float32  `json:"embedding,omitempty"` // omitted when nil
	Score            float64    `json:"score"`
	Anchors          []string   `json:"anchors,omitempty"`
	CreatedAt        *time.Time `json:"created_at,omitempty"`
	LastAccessedAt   *time.Time `json:"last_accessed_at,omitempty"`
	LastReinforcedAt *time.Time `json:"last_reinforced_at,omitempty"`
	ArchivedAt       *time.Time `json:"archived_at,omitempty"`
	AccessCount      *int64     `json:"access_count,omitempty"`
	StabilityScore   *float64   `json:"stability_score,omitempty"`
	IsArchived       *bool      `json:"is_archived,omitempty"`
	SimScore         *float64   `json:"sim_score,omitempty"`
	DenseScore       *float64   `json:"dense_score,omitempty"`
	BM25Score        *float64   `json:"bm25_score,omitempty"`
	FrequencyScore   *float64   `json:"frequency_score,omitempty"`
	RecencyScore     *float64   `json:"recency_score,omitempty"`
	AnchorScore      *float64   `json:"anchor_score,omitempty"`
	HopDistance      *int       `json:"hop_distance,omitempty"`
}

// ToDTO converts a ScoredNode to a ScoredNodeDTO, omitting Embedding unless includeEmbeddings is true.
func (sn ScoredNode) ToDTO(includeEmbeddings bool) ScoredNodeDTO {
	dto := ScoredNodeDTO{
		ID:             sn.ID,
		EntityType:     sn.EntityType,
		Label:          sn.Label,
		Summary:        sn.Summary,
		Score:          sn.Score,
		Anchors:        sn.Anchors,
		CreatedAt:      &sn.CreatedAt,
		LastAccessedAt: &sn.LastAccessedAt,
		ArchivedAt:     sn.ArchivedAt,
	}
	if sn.AccessCount > 0 {
		dto.AccessCount = &sn.AccessCount
	}
	if sn.StabilityScore > 0 {
		dto.StabilityScore = &sn.StabilityScore
	}
	if sn.IsArchived {
		dto.IsArchived = &sn.IsArchived
	}
	if sn.SimScore > 0 {
		dto.SimScore = &sn.SimScore
	}
	if sn.DenseScore != nil {
		dto.DenseScore = sn.DenseScore
	}
	if sn.BM25Score != nil {
		dto.BM25Score = sn.BM25Score
	}
	if sn.FrequencyScore > 0 {
		dto.FrequencyScore = &sn.FrequencyScore
	}
	if sn.RecencyScore > 0 {
		dto.RecencyScore = &sn.RecencyScore
	}
	if sn.AnchorScore > 0 {
		dto.AnchorScore = &sn.AnchorScore
	}
	if sn.HopDistance > 0 {
		dto.HopDistance = &sn.HopDistance
	}
	if !sn.LastReinforcedAt.IsZero() {
		dto.LastReinforcedAt = &sn.LastReinforcedAt
	}
	if includeEmbeddings && len(sn.Embedding) > 0 {
		dto.Embedding = sn.Embedding
	}
	return dto
}

// Project extracts only the requested fields into a map for JSON serialisation.
func (sn ScoredNode) Project(fields []string) map[string]any {
	out := make(map[string]any, len(fields))
	for _, rawField := range fields {
		f := strings.ToLower(strings.TrimSpace(rawField))
		switch f {
		case "id":
			out["id"] = sn.ID
		case "entity_type":
			out["entity_type"] = sn.EntityType
		case "label":
			out["label"] = sn.Label
		case "summary":
			out["summary"] = sn.Summary
		case "embedding":
			if sn.Embedding != nil {
				out["embedding"] = sn.Embedding
			} else {
				out["embedding"] = []float32{}
			}
		case "anchors":
			out["anchors"] = sn.Anchors
		case "created_at":
			out["created_at"] = sn.CreatedAt
		case "last_accessed_at":
			out["last_accessed_at"] = sn.LastAccessedAt
		case "last_reinforced_at":
			if !sn.LastReinforcedAt.IsZero() {
				out["last_reinforced_at"] = sn.LastReinforcedAt
			}
		case "archived_at":
			if sn.ArchivedAt != nil {
				out["archived_at"] = sn.ArchivedAt
			}
		case "access_count":
			out["access_count"] = sn.AccessCount
		case "stability_score":
			out["stability_score"] = sn.StabilityScore
		case "is_archived":
			out["is_archived"] = sn.IsArchived
		case "score":
			out["score"] = sn.Score
		case "sim_score":
			out["sim_score"] = sn.SimScore
		case "dense_score":
			if sn.DenseScore != nil {
				out["dense_score"] = *sn.DenseScore
			}
		case "bm25_score":
			if sn.BM25Score != nil {
				out["bm25_score"] = *sn.BM25Score
			}
		case "frequency_score":
			out["frequency_score"] = sn.FrequencyScore
		case "recency_score":
			out["recency_score"] = sn.RecencyScore
		case "anchor_score":
			out["anchor_score"] = sn.AnchorScore
		case "hop_distance":
			out["hop_distance"] = sn.HopDistance
		}
	}
	return out
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
	Query             string    `json:"query,omitempty"`
	Embedding         []float32 `json:"embedding,omitempty"`
	EntityID          string    `json:"entity_id,omitempty"`
	TopK              int       `json:"top_k,omitempty"`
	Alpha             float64   `json:"alpha,omitempty"`         // Weight for semantic similarity (default 0.6)
	Beta              float64   `json:"beta,omitempty"`          // Weight for access count frequency (default 0.2)
	Gamma             float64   `json:"gamma,omitempty"`         // Weight for recency decay (default 0.2)
	HybridAlpha       *float64  `json:"hybrid_alpha,omitempty"`  // Balance between dense semantic (1.0) and BM25 lexical (0.0), default 0.65
	Mode              string    `json:"mode,omitempty"`          // "hybrid" | "dense" | "bm25" (default: "hybrid")
	ExpandNeighbours  *bool     `json:"expand_neighbours,omitempty"` // Explicit toggle (default: false)
	ExpandHops        int       `json:"expand_hops,omitempty"`       // 0 = direct only, 1 = 1-hop expansion
	MinEdgeWeight     *float64  `json:"min_edge_weight,omitempty"`   // Minimum edge weight threshold (default: 0.60)
	TraverseRelations []string  `json:"traverse_relations,omitempty"` // Allowed edge relation types (e.g. ["subgoal_of", "depends_on"])
	AttenuationFactor *float64  `json:"attenuation_factor,omitempty"` // Neighbour score boost multiplier (default: 0.35)
	Anchors           []string  `json:"anchors,omitempty"`       // Target anchor tags (e.g. ["#project:kestrel"])
	AnchorMode        string    `json:"anchor_mode,omitempty"`   // "boost" | "filter" (default: "boost")
	AnchorWeight      float64   `json:"anchor_weight,omitempty"` // Weight for anchor bonus w_anc (default 1.0)
	IncludeEmbeddings bool      `json:"include_embeddings,omitempty"`
	Fields            []string  `json:"fields,omitempty"`
}

// ShouldExpandNeighbours returns true if ExpandNeighbours is explicitly true,
// or if ExpandNeighbours == nil and ExpandHops > 0. (Default: false when both are omitted).
func (r *RecallRequest) ShouldExpandNeighbours() bool {
	if r.ExpandNeighbours != nil {
		return *r.ExpandNeighbours
	}
	return r.ExpandHops > 0
}

// GetMinEdgeWeight returns the configured minimum edge weight threshold or default (0.60).
func (r *RecallRequest) GetMinEdgeWeight(defaultWeight float64) float64 {
	if r.MinEdgeWeight != nil {
		return *r.MinEdgeWeight
	}
	if defaultWeight > 0 {
		return defaultWeight
	}
	return 0.60
}

// GetAttenuationFactor returns the configured attenuation factor or default (0.35).
func (r *RecallRequest) GetAttenuationFactor(defaultAttn float64) float64 {
	if r.AttenuationFactor != nil {
		return *r.AttenuationFactor
	}
	if defaultAttn > 0 {
		return defaultAttn
	}
	return 0.35
}

// GetHybridAlpha returns the configured hybrid alpha weight (defaulting to defaultAlpha if nil).
func (r *RecallRequest) GetHybridAlpha(defaultAlpha float64) float64 {
	if r.HybridAlpha != nil {
		return *r.HybridAlpha
	}
	return defaultAlpha
}

// UnmarshalJSON implements custom JSON deserialization for RecallRequest
// to support fields and traverse_relations passed as either a JSON array or comma-separated string.
func (r *RecallRequest) UnmarshalJSON(data []byte) error {
	type Alias RecallRequest
	aux := struct {
		Fields            any `json:"fields,omitempty"`
		TraverseRelations any `json:"traverse_relations,omitempty"`
		*Alias
	}{
		Alias: (*Alias)(r),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	if aux.Fields != nil {
		switch v := aux.Fields.(type) {
		case string:
			for _, f := range strings.Split(v, ",") {
				clean := strings.TrimSpace(f)
				if clean != "" {
					r.Fields = append(r.Fields, clean)
				}
			}
		case []any:
			for _, item := range v {
				if s, ok := item.(string); ok {
					for _, f := range strings.Split(s, ",") {
						clean := strings.TrimSpace(f)
						if clean != "" {
							r.Fields = append(r.Fields, clean)
						}
					}
				}
			}
		case []string:
			for _, s := range v {
				for _, f := range strings.Split(s, ",") {
					clean := strings.TrimSpace(f)
					if clean != "" {
						r.Fields = append(r.Fields, clean)
					}
				}
			}
		}
	}

	if aux.TraverseRelations != nil {
		switch v := aux.TraverseRelations.(type) {
		case string:
			for _, rel := range strings.Split(v, ",") {
				clean := strings.TrimSpace(rel)
				if clean != "" {
					r.TraverseRelations = append(r.TraverseRelations, clean)
				}
			}
		case []any:
			for _, item := range v {
				if s, ok := item.(string); ok {
					for _, rel := range strings.Split(s, ",") {
						clean := strings.TrimSpace(rel)
						if clean != "" {
							r.TraverseRelations = append(r.TraverseRelations, clean)
						}
					}
				}
			}
		case []string:
			for _, s := range v {
				for _, rel := range strings.Split(s, ",") {
					clean := strings.TrimSpace(rel)
					if clean != "" {
						r.TraverseRelations = append(r.TraverseRelations, clean)
					}
				}
			}
		}
	}
	return nil
}

// RecallResponse returns ranked contextual nodes and their relational subgraph.
type RecallResponse struct {
	Nodes          []ScoredNode     `json:"nodes"`
	Edges          []Edge           `json:"edges,omitempty"`
	QueryLatencyMS float64          `json:"query_latency_ms"`
	ProjectedNodes []map[string]any `json:"-"`
}

// MarshalJSON customises JSON serialisation when field projection is active.
func (r RecallResponse) MarshalJSON() ([]byte, error) {
	if len(r.ProjectedNodes) > 0 {
		type Alias RecallResponse
		return json.Marshal(&struct {
			Nodes []map[string]any `json:"nodes"`
			*Alias
		}{
			Nodes: r.ProjectedNodes,
			Alias: (*Alias)(&r),
		})
	}
	type Alias RecallResponse
	return json.Marshal((*Alias)(&r))
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

// EmbeddingEngineHealth reports status and configuration of the dense semantic embedding client.
type EmbeddingEngineHealth struct {
	Enabled   bool   `json:"enabled"`
	Status    string `json:"status"` // "reachable", "degraded", "offline"
	URL       string `json:"url"`
	Dimension int    `json:"dimension"`
}

// HealthResponse reports service status and cluster node metadata.
type HealthResponse struct {
	Status          string                 `json:"status"`
	Node            string                 `json:"node,omitempty"`
	Port            int                    `json:"port,omitempty"`
	Service         string                 `json:"service,omitempty"`
	UptimeSeconds   int64                  `json:"uptime_seconds,omitempty"`
	NodeCount       int64                  `json:"node_count"`
	EdgeCount       int64                  `json:"edge_count"`
	EmbeddingEngine *EmbeddingEngineHealth `json:"embedding_engine,omitempty"`
}
