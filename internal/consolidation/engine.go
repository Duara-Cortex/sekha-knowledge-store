package consolidation

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

// Engine orchestrates the periodic background consolidation cycle, Hebbian reinforcement,
// mathematical recency decay, and soft-archiving routines.
type Engine struct {
	store     store.Store
	extractor *Extractor
	fusion    *FusionEngine
	config    model.DecayConfig
	mu        sync.RWMutex
	stats     model.ConsolidationStats
}

// NewEngine initialises the consolidation engine with SQLite storage and decay parameters.
func NewEngine(s store.Store, cfg model.DecayConfig) *Engine {
	extractor := NewExtractor(64)
	fusion := NewFusionEngine(s, cfg)

	return &Engine{
		store:     s,
		extractor: extractor,
		fusion:    fusion,
		config:    cfg,
		stats: model.ConsolidationStats{
			LastCycleAt: time.Now().UTC(),
		},
	}
}

// IngestTrace queues an episodic trace from Node 2 and optionally triggers inline consolidation.
func (e *Engine) IngestTrace(ctx context.Context, req model.ConsolidateRequest) (*model.ConsolidateResponse, error) {
	trace := model.EpisodicTrace{
		ID:               req.TraceID,
		SessionID:        req.SessionID,
		TaskGoal:         req.TaskGoal,
		Outcome:          req.Outcome,
		SensoryContext:   req.SensoryContext,
		Trajectory:       req.Trajectory,
		CandidateActions: req.CandidateActions,
		CreatedAt:        time.Now().UTC(),
	}

	if trace.TaskGoal == "" && req.ActiveGoal != "" {
		trace.TaskGoal = req.ActiveGoal
	}
	if trace.Outcome == "" && req.Status != "" {
		trace.Outcome = req.Status
	}

	if err := e.store.QueueTrace(ctx, trace); err != nil {
		return nil, fmt.Errorf("failed queueing episodic trace: %w", err)
	}

	resp := &model.ConsolidateResponse{
		Status:      "accepted",
		TraceID:     trace.ID,
		Message:     "Episodic deliberation trace queued for background consolidation",
		Synchronous: req.Synchronous,
	}

	// If synchronous execution requested, immediately fuse this trace
	if req.Synchronous {
		now := time.Now().UTC()
		extracted := e.extractor.Extract(trace)
		fusionRes, err := e.fusion.Fuse(ctx, extracted, now)
		if err != nil {
			return nil, fmt.Errorf("synchronous fusion failed: %w", err)
		}

		_ = e.store.MarkTraceConsolidated(ctx, trace.ID, now)

		resp.Status = "consolidated"
		resp.Message = "Episodic trace consolidated synchronously into knowledge graph"
		resp.EntitiesExtracted = len(extracted.Entities)
		resp.NodesFused = fusionRes.EntitiesFused + fusionRes.EntitiesCreated
		resp.EdgesReinforced = fusionRes.EdgesReinforced
	}

	return resp, nil
}

// RunConsolidationCycle executes a complete consolidation batch:
// 1. Ingests and fuses unconsolidated episodic deliberation traces.
// 2. Applies Hebbian reinforcement to co-activated pathways.
// 3. Evaluates exponential recency decay W(t) = W_0 * exp(-lambda * delta_t).
// 4. Soft-archives decaying nodes below omega_prune and prunes weak edges.
// 5. Updates retention and growth telemetry metrics.
func (e *Engine) RunConsolidationCycle(ctx context.Context, refTime time.Time) (*model.ConsolidationTriggerResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	start := time.Now()
	if refTime.IsZero() {
		refTime = start.UTC()
	}

	// 1. Fetch pending unconsolidated traces
	pendingTraces, err := e.store.GetPendingTraces(ctx, e.config.BatchSize)
	if err != nil {
		return nil, fmt.Errorf("failed retrieving pending traces: %w", err)
	}

	totalEntitiesExtracted := 0
	totalNodesFused := 0
	totalEdgesReinforced := 0

	// 2. Fuse pending traces into knowledge graph
	for _, trace := range pendingTraces {
		extracted := e.extractor.Extract(trace)
		totalEntitiesExtracted += len(extracted.Entities)

		traceTime := trace.CreatedAt
		if traceTime.IsZero() {
			traceTime = refTime
		}

		fusionRes, err := e.fusion.Fuse(ctx, extracted, traceTime)
		if err != nil {
			log.Printf("[Consolidation] Error fusing trace %s: %v", trace.ID, err)
			continue
		}

		totalNodesFused += (fusionRes.EntitiesFused + fusionRes.EntitiesCreated)
		totalEdgesReinforced += fusionRes.EdgesReinforced

		if err := e.store.MarkTraceConsolidated(ctx, trace.ID, refTime); err != nil {
			log.Printf("[Consolidation] Error marking trace %s consolidated: %v", trace.ID, err)
		}
	}

	// 3. Evaluate exponential recency decay and soft-archival / pruning
	nodesDecayed, nodesArchived, edgesPruned, err := e.store.ApplyDecayAndPrune(ctx, e.config, refTime)
	if err != nil {
		return nil, fmt.Errorf("failed executing decay and prune cycle: %w", err)
	}

	// 4. Collect updated telemetry statistics
	storeStats, err := e.store.GetConsolidationStats(ctx)
	if err == nil && storeStats != nil {
		e.stats = *storeStats
	}

	duration := float64(time.Since(start).Microseconds()) / 1000.0
	e.stats.TotalCycles++
	e.stats.LastCycleAt = refTime
	e.stats.LastCycleDurationMS = duration
	e.stats.PrunedEdges += int64(edgesPruned)

	log.Printf("[Consolidation Cycle #%d] Traces: %d | Fused Nodes: %d | Reinforced Edges: %d | Decayed: %d | Archived: %d | Pruned Edges: %d | Duration: %.2fms",
		e.stats.TotalCycles, len(pendingTraces), totalNodesFused, totalEdgesReinforced, nodesDecayed, nodesArchived, edgesPruned, duration)

	return &model.ConsolidationTriggerResponse{
		Status:          "success",
		Message:         fmt.Sprintf("Consolidation cycle #%d completed successfully", e.stats.TotalCycles),
		DurationMS:      duration,
		TracesFused:     len(pendingTraces),
		NodesDecayed:    nodesDecayed,
		NodesArchived:   nodesArchived,
		EdgesReinforced: totalEdgesReinforced,
		EdgesPruned:     edgesPruned,
		Stats:           e.stats,
	}, nil
}

// GetStats returns the latest telemetry statistics for the consolidation subsystem.
func (e *Engine) GetStats(ctx context.Context) (model.ConsolidationStats, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	// Refresh live counts from database
	liveStats, err := e.store.GetConsolidationStats(ctx)
	if err == nil && liveStats != nil {
		liveStats.TotalCycles = e.stats.TotalCycles
		liveStats.LastCycleAt = e.stats.LastCycleAt
		liveStats.LastCycleDurationMS = e.stats.LastCycleDurationMS
		return *liveStats, nil
	}
	return e.stats, nil
}

// StartScheduler launches the periodic background consolidation daemon ticker loop.
func (e *Engine) StartScheduler(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		interval = 15 * time.Minute
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	log.Printf("[Consolidation Daemon] Periodic consolidation cycle initialised (Interval: %s)", interval)

	for {
		select {
		case <-ctx.Done():
			log.Println("[Consolidation Daemon] Scheduler shutting down gracefully...")
			return
		case t := <-ticker.C:
			cycleCtx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			if _, err := e.RunConsolidationCycle(cycleCtx, t.UTC()); err != nil {
				log.Printf("[Consolidation Daemon ERROR] Consolidation cycle failed: %v", err)
			}
			cancel()
		}
	}
}
