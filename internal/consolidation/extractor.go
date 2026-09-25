package consolidation

import (
	"crypto/sha256"
	"fmt"
	"regexp"
	"strings"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/embedding"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
)

// Common stop words filtered out during entity extraction.
var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "in": true, "on": true, "at": true,
	"to": true, "for": true, "of": true, "with": true, "by": true, "from": true,
	"up": true, "about": true, "into": true, "over": true, "after": true,
	"and": true, "or": true, "but": true, "so": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true, "did": true,
	"this": true, "that": true, "these": true, "those": true, "it": true, "its": true,
	"we": true, "they": true, "them": true, "their": true, "our": true, "ours": true,
}

var tokenRegexp = regexp.MustCompile(`[a-zA-Z0-9_\-\.:]{3,}`)

// Extractor analyses completed deliberation traces and extracts salient concepts,
// decisions, causal relationships, and outcome ratings.
type Extractor struct {
	vectorDims int
}

// NewExtractor initialises a trace extractor producing deterministic normalised embeddings.
func NewExtractor(vectorDims int) *Extractor {
	if vectorDims <= 0 {
		vectorDims = model.DefaultVectorDim
	}
	return &Extractor{vectorDims: vectorDims}
}

// ExtractionResult bundles all derived graph artifacts from an episodic deliberation trace.
type ExtractionResult struct {
	Entities []model.ExtractedEntity
	Edges    []model.ExtractedRelation
	Outcome  string
	Salience float64
	Anchors  []string
	IsSecret bool
}

// Extract processes a single episodic deliberation trace into graph entities and causal relationships.
func (e *Extractor) Extract(trace model.EpisodicTrace) ExtractionResult {
	res := ExtractionResult{
		Entities: make([]model.ExtractedEntity, 0),
		Edges:    make([]model.ExtractedRelation, 0),
		Anchors:  trace.Anchors,
		IsSecret: trace.IsSecret,
	}

	// 1. Determine resolved outcome and salience multiplier
	outcome := strings.ToLower(strings.TrimSpace(trace.Outcome))
	switch outcome {
	case model.OutcomeSuccess, "completed", "ok", "ready":
		res.Outcome = model.OutcomeSuccess
		res.Salience = 1.0
	case model.OutcomeFailure, "error", "failed":
		res.Outcome = model.OutcomeFailure
		res.Salience = 0.1
	default:
		res.Outcome = model.OutcomeNeutral
		res.Salience = 0.5
	}

	seenLabels := make(map[string]string) // label -> entityID

	// 2. Extract Primary Task Goal entity
	goalText := strings.TrimSpace(trace.TaskGoal)
	if goalText == "" {
		goalText = fmt.Sprintf("Episodic Task Session %s", trace.SessionID)
	}

	goalID := fmt.Sprintf("task-%x", sha256.Sum256([]byte(goalText)))[:16]
	goalSummary := fmt.Sprintf("Episodic task goal for deliberation session %s with outcome %s", trace.SessionID, res.Outcome)
	goalEntity := model.ExtractedEntity{
		ID:              goalID,
		EntityType:      "task_goal",
		Label:           goalText,
		Summary:         goalSummary,
		Embedding:       e.generateEmbedding(goalText),
		Salience:        res.Salience,
		ImportanceScore: computeImportance("task_goal", goalText, goalSummary, trace.Anchors),
		IsSecret:        trace.IsSecret,
	}
	res.Entities = append(res.Entities, goalEntity)
	seenLabels[strings.ToLower(goalText)] = goalID

	// 3. Extract Trajectory Steps & Decisions
	var prevStepID string
	for _, step := range trace.Trajectory {
		stepSummary := strings.TrimSpace(step.Thought)
		if stepSummary == "" {
			stepSummary = strings.TrimSpace(step.Action)
		}
		if stepSummary == "" {
			continue
		}

		stepKey := fmt.Sprintf("%s-step-%d-%s", trace.SessionID, step.StepIndex, step.Action)
		stepID := fmt.Sprintf("dec-%x", sha256.Sum256([]byte(stepKey)))[:16]
		stepLabel := fmt.Sprintf("Step %d: %s", step.StepIndex, truncateString(step.Action, 48))

		stepEntity := model.ExtractedEntity{
			ID:              stepID,
			EntityType:      "decision",
			Label:           stepLabel,
			Summary:         stepSummary,
			Embedding:       e.generateEmbedding(stepSummary),
			Salience:        res.Salience * 0.85,
			ImportanceScore: computeImportance("decision", stepLabel, stepSummary, trace.Anchors),
			IsSecret:        step.IsSecret || trace.IsSecret,
		}
		res.Entities = append(res.Entities, stepEntity)

		// Edge: Decision -> Subgoal of Task Goal
		res.Edges = append(res.Edges, model.ExtractedRelation{
			SourceID:     stepID,
			TargetID:     goalID,
			RelationType: "subgoal_of",
			Weight:       1.0 * res.Salience,
		})

		// Causal temporal link: Step i -> Step i+1
		if prevStepID != "" {
			res.Edges = append(res.Edges, model.ExtractedRelation{
				SourceID:     prevStepID,
				TargetID:     stepID,
				RelationType: "precedes",
				Weight:       0.8 * res.Salience,
			})
		}
		prevStepID = stepID

		// Extract concept keywords from thought and observation
		textPool := fmt.Sprintf("%s %s %s", step.Thought, step.Action, step.Observation)
		keywords := e.extractKeywords(textPool)
		for _, kw := range keywords {
			lowerKw := strings.ToLower(kw)
			kwID, exists := seenLabels[lowerKw]
			if !exists {
				kwID = fmt.Sprintf("concept-%x", sha256.Sum256([]byte(lowerKw)))[:16]
				seenLabels[lowerKw] = kwID
				conceptSummary := fmt.Sprintf("Salient concept extracted from episodic deliberation: %s", kw)

				res.Entities = append(res.Entities, model.ExtractedEntity{
					ID:              kwID,
					EntityType:      "concept",
					Label:           kw,
					Summary:         conceptSummary,
					Embedding:       e.generateEmbedding(kw),
					Salience:        res.Salience * 0.7,
					ImportanceScore: computeImportance("concept", kw, conceptSummary, trace.Anchors),
					IsSecret:        step.IsSecret || trace.IsSecret,
				})
			}

			// Co-activation link between decision and concept
			res.Edges = append(res.Edges, model.ExtractedRelation{
				SourceID:     stepID,
				TargetID:     kwID,
				RelationType: "associates_with",
				Weight:       0.6 * res.Salience,
			})
		}
	}

	// 4. Extract High-Salience Sensory Context
	for _, chunk := range trace.SensoryContext {
		if chunk.Salience < 0.3 || strings.TrimSpace(chunk.Text) == "" {
			continue
		}
		chunkID := fmt.Sprintf("sensory-%x", sha256.Sum256([]byte(chunk.Text)))[:16]
		chunkLabel := truncateString(chunk.Text, 40)
		chunkEntity := model.ExtractedEntity{
			ID:              chunkID,
			EntityType:      "sensory_fact",
			Label:           chunkLabel,
			Summary:         chunk.Text,
			Embedding:       e.generateEmbedding(chunk.Text),
			Salience:        chunk.Salience * res.Salience,
			ImportanceScore: computeImportance("sensory_fact", chunkLabel, chunk.Text, trace.Anchors),
			IsSecret:        chunk.IsSecret || trace.IsSecret,
		}
		res.Entities = append(res.Entities, chunkEntity)

		res.Edges = append(res.Edges, model.ExtractedRelation{
			SourceID:     chunkID,
			TargetID:     goalID,
			RelationType: "context_for",
			Weight:       chunk.Salience * res.Salience,
		})
	}

	return res
}

// ComputeImportance calculates an intrinsic importance score in [0.0, 1.0] based on entity type,
// semantic indicators in label and summary, and anchor tags.
func ComputeImportance(entityType, label, summary string, anchors []string) float64 {
	return computeImportance(entityType, label, summary, anchors)
}

// computeImportance assigns an intrinsic importance score based on heuristics:
// - System config / architecture rule: 0.90
// - Task goal: 0.85
// - Decision / procedure: 0.75
// - Anchor hubs (+0.15 boost, capped at 1.0, >= 0.85)
// - Generic sensory / telemetry fact: 0.40 - 0.50 (0.45)
// - Transient unanchored noise: 0.20 - 0.30 (0.25)
func computeImportance(entityType, label, summary string, anchors []string) float64 {
	eType := strings.ToLower(strings.TrimSpace(entityType))
	lowerLabel := strings.ToLower(label)
	lowerSummary := strings.ToLower(summary)

	var score float64

	// 1. System configs, architecture rules: >= 0.85 (default 0.90)
	isConfigOrRule := eType == "system_config" || eType == "architecture_rule" || eType == "config" || eType == "rule" ||
		strings.Contains(lowerLabel, "config") || strings.Contains(lowerLabel, "architecture") ||
		strings.Contains(lowerLabel, "api_key") || strings.Contains(lowerLabel, "ingest_port") ||
		strings.Contains(lowerLabel, "network bounds") || strings.Contains(lowerLabel, "master config") ||
		strings.Contains(lowerSummary, "system config") || strings.Contains(lowerSummary, "architecture rule")

	// 2. Task goals: >= 0.85
	isGoal := eType == "task_goal" || strings.Contains(lowerLabel, "task goal")

	// 3. Decisions and procedures: 0.70 - 0.85 (default 0.75)
	isDecisionOrProc := eType == "decision" || eType == "procedure" ||
		strings.Contains(lowerLabel, "decision") || strings.Contains(lowerLabel, "procedure") ||
		strings.HasPrefix(lowerLabel, "step ")

	// 4. Transient unanchored noise: 0.20 - 0.30 (default 0.25)
	isTransientNoise := strings.Contains(lowerLabel, "jitter") || strings.Contains(lowerLabel, "transient") ||
		strings.Contains(lowerLabel, "noise") || strings.Contains(lowerLabel, "ephemeral") ||
		strings.Contains(lowerLabel, "speculative") || strings.Contains(lowerLabel, "debug log") ||
		strings.Contains(lowerSummary, "transient") || strings.Contains(lowerSummary, "ephemeral") ||
		strings.Contains(lowerSummary, "jitter")

	// 5. Generic sensory / telemetry facts: 0.40 - 0.50 (default 0.45)
	isSensoryOrTelemetry := eType == "sensory_fact" || eType == "telemetry" || eType == "telemetry_chunk" ||
		eType == "telemetry_rule" || eType == "episodic_event" ||
		strings.Contains(lowerLabel, "telemetry") || strings.Contains(lowerLabel, "sensor") ||
		strings.Contains(lowerSummary, "telemetry")

	if isConfigOrRule {
		score = 0.90
	} else if isGoal {
		score = 0.85
	} else if isDecisionOrProc {
		score = 0.75
	} else if isTransientNoise {
		score = 0.25
	} else if isSensoryOrTelemetry {
		score = 0.45
	} else {
		score = 0.50
	}

	// Anchor tags boost: +0.15 for nodes with anchor tags (cap at 1.0)
	if len(anchors) > 0 {
		score += 0.15
		if score > 1.0 {
			score = 1.0
		}
	}

	return score
}

// extractKeywords extracts distinctive tokens from a block of text.
func (e *Extractor) extractKeywords(text string) []string {
	matches := tokenRegexp.FindAllString(text, -1)
	seen := make(map[string]bool)
	var keywords []string

	for _, token := range matches {
		clean := strings.ToLower(strings.Trim(token, ".,:;()[]\"'"))
		if len(clean) < 3 || stopWords[clean] || seen[clean] {
			continue
		}
		seen[clean] = true
		keywords = append(keywords, clean)
		if len(keywords) >= 8 { // Bounded keyword pool per step to prevent graph bloat
			break
		}
	}
	return keywords
}

// generateEmbedding synthesises a deterministic, unit-normalised float32 semantic vector
// using subword, stem, and token feature projections.
func (e *Extractor) generateEmbedding(text string) []float32 {
	return embedding.Generate(text, e.vectorDims)
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
