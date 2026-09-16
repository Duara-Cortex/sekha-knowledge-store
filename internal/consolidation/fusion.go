package consolidation

import (
	"context"
	"fmt"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

// FusionEngine performs graph deduplication, entity reconciliation, and relational fusion
// against existing SQLite knowledge nodes.
type FusionEngine struct {
	store  store.Store
	config model.DecayConfig
}

// NewFusionEngine initialises a new graph fusion and deduplication engine.
func NewFusionEngine(s store.Store, cfg model.DecayConfig) *FusionEngine {
	return &FusionEngine{
		store:  s,
		config: cfg,
	}
}

// FusionResult captures the outcomes of entity reconciliation and edge reinforcement.
type FusionResult struct {
	EntitiesFused    int
	EntitiesCreated  int
	EdgesReinforced  int
	EntityIDMappings map[string]string // trace entity ID -> canonical persistent node ID
	CreatedNodes     []model.Node
}

// Fuse assimilates extracted entities and relationships into the persistent knowledge store.
func (f *FusionEngine) Fuse(ctx context.Context, extraction ExtractionResult, refTime time.Time) (*FusionResult, error) {
	if refTime.IsZero() {
		refTime = time.Now().UTC()
	}

	result := &FusionResult{
		EntityIDMappings: make(map[string]string),
	}

	nodesToInsert := make([]model.Node, 0)

	// 1. Reconcile entities against existing persistent nodes
	for _, extracted := range extraction.Entities {
		existing, err := f.store.FindMatchingNode(ctx, extracted.Label, extracted.EntityType)
		if err != nil {
			return nil, fmt.Errorf("error checking existing node for %s: %w", extracted.Label, err)
		}

		if existing != nil {
			// Entity deduplication match: preserve existing persistent node ID
			result.EntityIDMappings[extracted.ID] = existing.ID

			// Hebbian stability reinforcement for recurring hub entities
			deltaStability := f.config.HebbianLearningRate * extracted.Salience
			if err := f.store.BoostNodeStability(ctx, existing.ID, deltaStability, refTime); err != nil {
				return nil, fmt.Errorf("failed boosting stability for node %s: %w", existing.ID, err)
			}
			result.EntitiesFused++
		} else {
			// Novel entity: allocate persistent knowledge vertex
			canonicalID := extracted.ID
			result.EntityIDMappings[extracted.ID] = canonicalID

			initialStability := 0.5 + (0.5 * extracted.Salience)
			if initialStability > 1.0 {
				initialStability = 1.0
			}

			nodesToInsert = append(nodesToInsert, model.Node{
				ID:               canonicalID,
				EntityType:       extracted.EntityType,
				Label:            extracted.Label,
				Summary:          extracted.Summary,
				Embedding:        extracted.Embedding,
				CreatedAt:        refTime,
				LastAccessedAt:   refTime,
				LastReinforcedAt: refTime,
				AccessCount:      1,
				StabilityScore:   initialStability,
				IsArchived:       false,
			})
			result.EntitiesCreated++
		}
	}

	// 2. Persist novel nodes
	if len(nodesToInsert) > 0 {
		if _, err := f.store.InsertNodes(ctx, nodesToInsert); err != nil {
			return nil, fmt.Errorf("failed persisting novel nodes during fusion: %w", err)
		}
		result.CreatedNodes = nodesToInsert
	}

	// 3. Reconcile and reinforce relational edges using canonical IDs
	for _, rel := range extraction.Edges {
		sourceID, srcFound := result.EntityIDMappings[rel.SourceID]
		if !srcFound {
			sourceID = rel.SourceID
		}
		targetID, tgtFound := result.EntityIDMappings[rel.TargetID]
		if !tgtFound {
			targetID = rel.TargetID
		}

		// Avoid self-loops during deduplicated edge synthesis
		if sourceID == targetID {
			continue
		}

		// Hebbian co-activation reinforcement: Delta W = eta * initial_weight * salience
		deltaW := f.config.HebbianLearningRate * rel.Weight * extraction.Salience
		if _, err := f.store.ReinforceEdge(ctx, sourceID, targetID, rel.RelationType, deltaW, f.config.MaxEdgeWeight, refTime); err != nil {
			return nil, fmt.Errorf("failed reinforcing relational edge %s->%s: %w", sourceID, targetID, err)
		}
		result.EdgesReinforced++
	}

	return result, nil
}
