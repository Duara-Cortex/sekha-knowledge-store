package consolidation

import (
	"crypto/sha256"
	"fmt"
	"math"
	"regexp"
	"sort"
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

// incidentConceptLimit bounds the concepts linked from each incident line in an execution trace.
const incidentConceptLimit = 6

var (
	// instanceSuffixRegexp matches per-instance numeric suffixes such as sample_17 or row-group-3.
	instanceSuffixRegexp = regexp.MustCompile(`([_\-]\d+)+$`)
	// numberWithUnitRegexp matches measured values such as 60.00c, 2500rpm or 1000mbps.
	numberWithUnitRegexp = regexp.MustCompile(`^[\d.:]+[a-z]{1,4}$`)
	// hexDigestRegexp matches checksums and hashes such as 98af12 or deadbeef01.
	hexDigestRegexp = regexp.MustCompile(`^[0-9a-f]*\d[0-9a-f]*$`)
	// logTimestampPrefix matches a leading log timestamp (12:00:01.000, 2026-09-25T13:03:20Z, [2026-09-25 13:03:20]).
	logTimestampPrefix = regexp.MustCompile(`^[\[(]?\d{2,4}[-/:.][\d\-/:.TZ+]*(?:[ T]\d{2}:\d{2}[\d:.,Z+\-]*)?[\])]?\s+`)
)

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

// Extract processes an episodic deliberation trace into graph entities and causal relationships,
// scaling dynamically with trace content and length without hardcoded entity ceilings.
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

	seenLabels := make(map[string]string) // key: lower(label)+"::"+entityType -> entityID

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
	seenLabels[strings.ToLower(goalText)+"::task_goal"] = goalID

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
		seenLabels[strings.ToLower(stepLabel)+"::decision"] = stepID

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

		// Extract concept keywords dynamically based on pool length
		textPool := fmt.Sprintf("%s %s %s", step.Thought, step.Action, step.Observation)
		keywords := e.extractKeywords(textPool, dynamicKeywordLimit(len(textPool)))
		for _, kw := range keywords {
			dedupKey := kw + "::concept"
			kwID, exists := seenLabels[dedupKey]
			if !exists {
				kwID = fmt.Sprintf("concept-%x", sha256.Sum256([]byte(kw)))[:16]
				seenLabels[dedupKey] = kwID
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

	// 4. Extract Candidate Actions
	for _, action := range trace.CandidateActions {
		actionType := strings.TrimSpace(action.Type)
		if actionType == "" {
			continue
		}
		actionKey := fmt.Sprintf("%s-act-%s-%s", trace.SessionID, action.ID, actionType)
		actID := fmt.Sprintf("act-%x", sha256.Sum256([]byte(actionKey)))[:16]
		actLabel := fmt.Sprintf("Action: %s", truncateString(actionType, 40))
		actSummary := fmt.Sprintf("Candidate deliberation action %s (committed: %t)", actionType, action.Committed)

		dedupKey := strings.ToLower(actLabel) + "::candidate_action"
		if _, exists := seenLabels[dedupKey]; !exists {
			seenLabels[dedupKey] = actID
			res.Entities = append(res.Entities, model.ExtractedEntity{
				ID:              actID,
				EntityType:      "candidate_action",
				Label:           actLabel,
				Summary:         actSummary,
				Embedding:       e.generateEmbedding(actSummary),
				Salience:        res.Salience * 0.8,
				ImportanceScore: computeImportance("candidate_action", actLabel, actSummary, trace.Anchors),
				IsSecret:        action.IsSecret || trace.IsSecret,
			})

			res.Edges = append(res.Edges, model.ExtractedRelation{
				SourceID:     actID,
				TargetID:     goalID,
				RelationType: "proposed_for",
				Weight:       0.75 * res.Salience,
			})

			// Extract concepts from action type and payload
			actionPool := actionType
			for k, v := range action.Payload {
				actionPool += fmt.Sprintf(" %s %v", k, v)
			}
			actKeywords := e.extractKeywords(actionPool, dynamicKeywordLimit(len(actionPool)))
			for _, kw := range actKeywords {
				cKey := kw + "::concept"
				kwID, cExists := seenLabels[cKey]
				if !cExists {
					kwID = fmt.Sprintf("concept-%x", sha256.Sum256([]byte(kw)))[:16]
					seenLabels[cKey] = kwID
					res.Entities = append(res.Entities, model.ExtractedEntity{
						ID:              kwID,
						EntityType:      "concept",
						Label:           kw,
						Summary:         fmt.Sprintf("Concept from action payload: %s", kw),
						Embedding:       e.generateEmbedding(kw),
						Salience:        res.Salience * 0.65,
						ImportanceScore: computeImportance("concept", kw, kw, trace.Anchors),
						IsSecret:        action.IsSecret || trace.IsSecret,
					})
				}
				res.Edges = append(res.Edges, model.ExtractedRelation{
					SourceID:     actID,
					TargetID:     kwID,
					RelationType: "associates_with",
					Weight:       0.55 * res.Salience,
				})
			}
		}
	}

	// 5. Extract High-Salience Sensory Context
	for _, chunk := range trace.SensoryContext {
		if chunk.Salience < 0.3 || strings.TrimSpace(chunk.Text) == "" {
			continue
		}
		chunkID := fmt.Sprintf("sensory-%x", sha256.Sum256([]byte(chunk.Text)))[:16]
		chunkLabel := truncateString(chunk.Text, 40)
		chunkDedup := strings.ToLower(chunkLabel) + "::sensory_fact"

		if _, exists := seenLabels[chunkDedup]; !exists {
			seenLabels[chunkDedup] = chunkID
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

			// Extract concepts from substantial sensory chunk text
			if len(chunk.Text) > 60 {
				sensoryKeywords := e.extractKeywords(chunk.Text, dynamicKeywordLimit(len(chunk.Text)))
				for _, kw := range sensoryKeywords {
					cKey := kw + "::concept"
					kwID, cExists := seenLabels[cKey]
					if !cExists {
						kwID = fmt.Sprintf("concept-%x", sha256.Sum256([]byte(kw)))[:16]
						seenLabels[cKey] = kwID
						res.Entities = append(res.Entities, model.ExtractedEntity{
							ID:              kwID,
							EntityType:      "concept",
							Label:           kw,
							Summary:         fmt.Sprintf("Sensory context concept: %s", kw),
							Embedding:       e.generateEmbedding(kw),
							Salience:        chunk.Salience * res.Salience * 0.7,
							ImportanceScore: computeImportance("concept", kw, kw, trace.Anchors),
							IsSecret:        chunk.IsSecret || trace.IsSecret,
						})
					}
					res.Edges = append(res.Edges, model.ExtractedRelation{
						SourceID:     chunkID,
						TargetID:     kwID,
						RelationType: "context_for",
						Weight:       0.5 * chunk.Salience * res.Salience,
					})
				}
			}
		}
	}

	// 6. Extract Execution Traces and Logs
	if execTrace := trace.GetExecutionTraceText(); strings.TrimSpace(execTrace) != "" {
		e.extractFromExecutionTrace(execTrace, goalID, &res, seenLabels)
	}

	return res
}

// extractFromExecutionTrace parses raw execution traces and logs into notable events and technical concepts,
// scaling dynamically with trace size and content.
func (e *Extractor) extractFromExecutionTrace(execTrace string, goalID string, res *ExtractionResult, seenLabels map[string]string) {
	lines := strings.Split(execTrace, "\n")
	var notableLines []string
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if len(trimmed) >= 8 {
			notableLines = append(notableLines, trimmed)
		}
	}

	traceLen := len(execTrace)
	var maxEvents int
	if traceLen < 3000 {
		maxEvents = 3
	} else if traceLen < 25000 {
		maxEvents = 10
	} else if traceLen < 100000 {
		maxEvents = 25
	} else {
		maxEvents = 40
	}

	// 1. Extract notable line events. Incident lines (errors, failures, timeouts) are always
	// selected first so they are never skipped by sampling; the remaining budget is filled by
	// stride-sampling the rest of the trace for telemetry checkpoints and phase transitions.
	// Repeated incidents (e.g. a retry loop) are collapsed first so they cannot exhaust the budget.
	var incidentIdx, otherIdx []int
	seenIncidents := make(map[string]bool)
	for i, line := range notableLines {
		if classifyTraceLine(strings.ToLower(line)) == "incident" {
			key := strings.ToLower(traceEventLabel(line))
			if !seenIncidents[key] {
				seenIncidents[key] = true
				incidentIdx = append(incidentIdx, i)
			}
		} else {
			otherIdx = append(otherIdx, i)
		}
	}
	selected := make([]int, 0, maxEvents)
	selected = append(selected, incidentIdx...)
	if remaining := maxEvents - len(selected); remaining > 0 && len(otherIdx) > 0 {
		stepSize := 1
		if len(otherIdx) > remaining {
			stepSize = len(otherIdx) / remaining
		}
		for j := 0; j < len(otherIdx); j += stepSize {
			selected = append(selected, otherIdx[j])
		}
	}

	eventsExtracted := 0
	for _, i := range selected {
		if eventsExtracted >= maxEvents {
			break
		}
		line := notableLines[i]
		eType := classifyTraceLine(strings.ToLower(line))

		labelSource := stripLogTimestamp(line)
		label := traceEventLabel(line)
		dedupKey := strings.ToLower(label) + "::" + eType
		if _, exists := seenLabels[dedupKey]; !exists {
			evID := fmt.Sprintf("evt-%x", sha256.Sum256([]byte(fmt.Sprintf("%s-%d", line, i))))[:16]
			seenLabels[dedupKey] = evID

			entity := model.ExtractedEntity{
				ID:              evID,
				EntityType:      eType,
				Label:           label,
				Summary:         line,
				Embedding:       e.generateEmbedding(line),
				Salience:        res.Salience * 0.8,
				ImportanceScore: computeImportance(eType, label, line, res.Anchors),
				IsSecret:        res.IsSecret,
			}
			res.Entities = append(res.Entities, entity)
			eventsExtracted++

			relType := "context_for"
			if eType == "procedure" {
				relType = "subgoal_of"
			}
			res.Edges = append(res.Edges, model.ExtractedRelation{
				SourceID:     evID,
				TargetID:     goalID,
				RelationType: relType,
				Weight:       0.7 * res.Salience,
			})

			// Incidents are rare but high-signal, so their terms are always linked as concepts
			// rather than left to compete with frequent routine tokens in the trace-wide ranking.
			if eType == "incident" {
				for _, kw := range e.extractKeywords(labelSource, incidentConceptLimit) {
					res.Edges = append(res.Edges, model.ExtractedRelation{
						SourceID:     evID,
						TargetID:     e.traceConcept(kw, res, seenLabels),
						RelationType: "associates_with",
						Weight:       0.6 * res.Salience,
					})
				}
			}
		}
	}

	// 2. Extract technical concepts across the execution trace dynamically
	maxConcepts := dynamicKeywordLimit(traceLen)
	if traceLen > 25000 {
		maxConcepts = int(float64(maxConcepts) * 1.5)
		if maxConcepts > 100 {
			maxConcepts = 100
		}
	}
	keywords := e.extractKeywords(execTrace, maxConcepts)
	for _, kw := range keywords {
		res.Edges = append(res.Edges, model.ExtractedRelation{
			SourceID:     goalID,
			TargetID:     e.traceConcept(kw, res, seenLabels),
			RelationType: "associates_with",
			Weight:       0.6 * res.Salience,
		})
	}
}

// traceConcept returns the entity ID for an execution trace concept, adding the entity on first sight.
func (e *Extractor) traceConcept(kw string, res *ExtractionResult, seenLabels map[string]string) string {
	dedupKey := kw + "::concept"
	if kwID, exists := seenLabels[dedupKey]; exists {
		return kwID
	}
	kwID := fmt.Sprintf("concept-%x", sha256.Sum256([]byte(kw)))[:16]
	seenLabels[dedupKey] = kwID
	conceptSummary := fmt.Sprintf("Execution trace concept: %s", kw)

	res.Entities = append(res.Entities, model.ExtractedEntity{
		ID:              kwID,
		EntityType:      "concept",
		Label:           kw,
		Summary:         conceptSummary,
		Embedding:       e.generateEmbedding(kw),
		Salience:        res.Salience * 0.7,
		ImportanceScore: computeImportance("concept", kw, conceptSummary, res.Anchors),
		IsSecret:        res.IsSecret,
	})
	return kwID
}

// dynamicKeywordLimit calculates a dynamically scaled keyword limit based on text character length.
func dynamicKeywordLimit(textLength int) int {
	if textLength <= 200 {
		return 8
	}
	limit := 8 + int(math.Sqrt(float64(textLength))*0.4)
	if limit > 80 {
		limit = 80
	}
	return limit
}

// stripLogTimestamp removes a leading timestamp so recurring events share a label across traces.
func stripLogTimestamp(line string) string {
	if stripped := strings.TrimSpace(logTimestampPrefix.ReplaceAllString(line, "")); stripped != "" {
		return stripped
	}
	return line
}

// traceEventLabel returns the node label for an execution trace line.
func traceEventLabel(line string) string {
	return truncateString(stripLogTimestamp(line), 48)
}

// classifyTraceLine assigns an entity type to a lowercased execution trace line.
func classifyTraceLine(lowerLine string) string {
	switch {
	case strings.Contains(lowerLine, "error") || strings.Contains(lowerLine, "fail") ||
		strings.Contains(lowerLine, "drop") || strings.Contains(lowerLine, "fatal") ||
		strings.Contains(lowerLine, "timeout") || strings.Contains(lowerLine, "panic"):
		return "incident"
	case strings.Contains(lowerLine, "temp") || strings.Contains(lowerLine, "fan") ||
		strings.Contains(lowerLine, "pwm") || strings.Contains(lowerLine, "rpm") ||
		strings.Contains(lowerLine, "packet") || strings.Contains(lowerLine, "parquet") ||
		strings.Contains(lowerLine, "crc") || strings.Contains(lowerLine, "telemetry") ||
		strings.Contains(lowerLine, "="):
		return "telemetry"
	case strings.Contains(lowerLine, "start") || strings.Contains(lowerLine, "init") ||
		strings.Contains(lowerLine, "connect") || strings.Contains(lowerLine, "complete") ||
		strings.Contains(lowerLine, "schema") || strings.Contains(lowerLine, "finish"):
		return "procedure"
	default:
		return "episodic_event"
	}
}

// isNoiseToken reports whether a cleaned token is a value rather than a concept:
// timestamps, IPs, bare numbers, number+unit readings (60.00c, 2500rpm) or hex digests.
func isNoiseToken(s string) bool {
	letters := 0
	for _, c := range s {
		if c >= 'a' && c <= 'z' {
			letters++
		}
	}
	if letters < 3 {
		return true
	}
	return numberWithUnitRegexp.MatchString(s) || hexDigestRegexp.MatchString(s)
}

// ComputeImportance calculates an intrinsic importance score in [0.0, 1.0] based on entity type,
// semantic indicators in label and summary, and anchor tags.
func ComputeImportance(entityType, label, summary string, anchors []string) float64 {
	return computeImportance(entityType, label, summary, anchors)
}

// computeImportance assigns an intrinsic importance score based on heuristics:
// - System config / architecture rule: 0.90
// - Task goal: 0.85
// - Decision / procedure / candidate action: 0.75
// - Incident / error state: 0.80
// - Anchor hubs (+0.15 boost, capped at 1.0, >= 0.85)
// - Generic sensory / telemetry fact: 0.40 - 0.50 (0.45)
// - Concept / episodic event: 0.50 - 0.60
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

	// 3. Incidents, errors: 0.80
	isIncident := eType == "incident" || eType == "error_state" ||
		strings.Contains(lowerLabel, "error") || strings.Contains(lowerLabel, "fatal") ||
		strings.Contains(lowerLabel, "incident")

	// 4. Decisions, procedures, actions: 0.70 - 0.85 (default 0.75)
	isDecisionOrProc := eType == "decision" || eType == "procedure" || eType == "candidate_action" || eType == "action" ||
		strings.Contains(lowerLabel, "decision") || strings.Contains(lowerLabel, "procedure") ||
		strings.HasPrefix(lowerLabel, "step ") || strings.HasPrefix(lowerLabel, "action:")

	// 5. Transient unanchored noise: 0.20 - 0.30 (default 0.25)
	isTransientNoise := strings.Contains(lowerLabel, "jitter") || strings.Contains(lowerLabel, "transient") ||
		strings.Contains(lowerLabel, "noise") || strings.Contains(lowerLabel, "ephemeral") ||
		strings.Contains(lowerLabel, "speculative") || strings.Contains(lowerLabel, "debug log") ||
		strings.Contains(lowerSummary, "transient") || strings.Contains(lowerSummary, "ephemeral") ||
		strings.Contains(lowerSummary, "jitter")

	// 6. Generic sensory / telemetry facts: 0.40 - 0.50 (default 0.45)
	isSensoryOrTelemetry := eType == "sensory_fact" || eType == "telemetry" || eType == "telemetry_chunk" ||
		eType == "telemetry_rule" ||
		strings.Contains(lowerLabel, "telemetry") || strings.Contains(lowerLabel, "sensor") ||
		strings.Contains(lowerSummary, "telemetry")

	if isConfigOrRule {
		score = 0.90
	} else if isGoal {
		score = 0.85
	} else if isIncident {
		score = 0.80
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

// extractKeywords extracts up to maxCount distinctive concept tokens from a block of text.
// Value-like tokens (timestamps, readings, digests) are discarded and per-instance numeric
// suffixes are folded (sample_17 -> sample). When more candidates exist than maxCount, the
// most frequent are kept, so long traces yield concepts representative of the whole text.
func (e *Extractor) extractKeywords(text string, maxCount int) []string {
	if maxCount <= 0 {
		maxCount = dynamicKeywordLimit(len(text))
	}
	matches := tokenRegexp.FindAllString(text, -1)
	counts := make(map[string]int)
	var ordered []string

	for _, token := range matches {
		clean := strings.ToLower(strings.Trim(token, ".,:;()[]\"'{}<>/\\|`=-_"))
		clean = instanceSuffixRegexp.ReplaceAllString(clean, "")
		if len(clean) < 3 || stopWords[clean] || isNoiseToken(clean) {
			continue
		}
		if counts[clean] == 0 {
			ordered = append(ordered, clean)
		}
		counts[clean]++
	}

	if len(ordered) <= maxCount {
		return ordered
	}

	ranked := make([]string, len(ordered))
	copy(ranked, ordered)
	sort.SliceStable(ranked, func(i, j int) bool { return counts[ranked[i]] > counts[ranked[j]] })
	keep := make(map[string]bool, maxCount)
	for _, kw := range ranked[:maxCount] {
		keep[kw] = true
	}

	keywords := make([]string, 0, maxCount)
	for _, kw := range ordered {
		if keep[kw] {
			keywords = append(keywords, kw)
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
