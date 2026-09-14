package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/consolidation"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

func main() {
	eventCount := flag.Int("events", 1000, "Number of synthetic episodic deliberation events to simulate")
	dbPath := flag.String("db", "test_consolidation_benchmark.db", "SQLite database file for stress testing")
	decayTau := flag.Duration("decay-tau", 72*time.Hour, "Decay half-life duration (tau)")
	pruneThreshold := flag.Float64("prune-threshold", 0.10, "Omega_prune retention threshold")
	edgePruneThreshold := flag.Float64("edge-prune-threshold", 0.05, "Edge prune threshold")
	gracePeriod := flag.Duration("grace-period", 168*time.Hour, "Grace period before archival (7 days)")
	hebbianRate := flag.Float64("hebbian-rate", 0.15, "Hebbian learning rate eta")
	maxWeight := flag.Float64("max-weight", 5.0, "Maximum edge weight saturation")
	flag.Parse()

	fmt.Println("=================================================================")
	fmt.Println("   Sekha Memory Consolidation & Decay Daemon Benchmark Suite     ")
	fmt.Println("   Node 1 (8GB RAM Pi 5) 1,000 Episodic Events Stress Test       ")
	fmt.Println("=================================================================")
	fmt.Printf(" Target Events:        %d synthetic episodic traces\n", *eventCount)
	fmt.Printf(" Database File:        %s\n", *dbPath)
	fmt.Printf(" Decay Half-Life (tau):%s\n", *decayTau)
	fmt.Printf(" Retention (omega):    %.2f\n", *pruneThreshold)
	fmt.Printf(" Edge Prune Threshold: %.2f\n", *edgePruneThreshold)
	fmt.Printf(" Inactivity Grace:     %s\n", *gracePeriod)
	fmt.Println("-----------------------------------------------------------------")

	// Clean up any stale benchmark database
	_ = os.Remove(*dbPath)
	_ = os.Remove(*dbPath + "-wal")
	_ = os.Remove(*dbPath + "-shm")
	defer func() {
		_ = os.Remove(*dbPath)
		_ = os.Remove(*dbPath + "-wal")
		_ = os.Remove(*dbPath + "-shm")
	}()

	sqliteStore, err := store.NewSQLiteStore(*dbPath)
	if err != nil {
		fmt.Printf("[FATAL] Failed to initialise SQLite store: %v\n", err)
		os.Exit(1)
	}
	defer sqliteStore.Close()

	decayCfg := model.DecayConfig{
		DecayHalfLife:         *decayTau,
		PruneThreshold:        *pruneThreshold,
		EdgePruneThreshold:    *edgePruneThreshold,
		InactivityGracePeriod: *gracePeriod,
		HebbianLearningRate:   *hebbianRate,
		MaxEdgeWeight:         *maxWeight,
		BatchSize:             500,
	}

	engine := consolidation.NewEngine(sqliteStore, decayCfg)
	ctx := context.Background()

	// 1. Define synthetic hub concepts vs transient noise templates
	hubConcepts := []string{
		"cluster_coordinator",
		"sensory_salience_filter",
		"working_memory_deliberation",
		"sqlite_knowledge_graph",
		"hebbian_reinforcement_engine",
		"associative_recall_vector",
		"arm64_llama_inference",
		"circular_ring_buffer",
	}

	rng := rand.New(rand.NewSource(42))
	baseTime := time.Now().UTC().Add(-14 * 24 * time.Hour) // Simulate events beginning 14 days ago

	fmt.Printf("[Step 1] Ingesting %d synthetic episodic deliberation traces...\n", *eventCount)
	ingestStart := time.Now()

	// 80% transient noise, 20% recurrent hub events
	for i := 0; i < *eventCount; i++ {
		isHub := (i % 5 == 0) // 20% recurrent hub
		eventTime := baseTime.Add(time.Duration(rng.Float64()*7.0*24.0) * time.Hour)

		var trace model.EpisodicTrace
		trace.ID = fmt.Sprintf("trace-%05d", i+1)
		trace.SessionID = fmt.Sprintf("sess-%05d", i+1)
		trace.CreatedAt = eventTime

		if isHub {
			// Select 2 recurrent hub concepts to co-activate
			c1 := hubConcepts[rng.Intn(len(hubConcepts))]
			c2 := hubConcepts[rng.Intn(len(hubConcepts))]
			for c2 == c1 {
				c2 = hubConcepts[rng.Intn(len(hubConcepts))]
			}

			trace.TaskGoal = fmt.Sprintf("Optimise co-activation between %s and %s", c1, c2)
			trace.Outcome = model.OutcomeSuccess
			trace.Trajectory = []model.TrajectoryStep{
				{
					StepIndex:   1,
					Thought:     fmt.Sprintf("Evaluating active association for %s", c1),
					Action:      fmt.Sprintf("activate_%s", c1),
					Observation: "State verified",
					Status:      "success",
					Timestamp:   eventTime,
				},
				{
					StepIndex:   2,
					Thought:     fmt.Sprintf("Synthesising pathway to %s", c2),
					Action:      fmt.Sprintf("connect_%s", c2),
					Observation: "Connection established",
					Status:      "success",
					Timestamp:   eventTime.Add(time.Second),
				},
			}
			trace.SensoryContext = []model.SensoryItem{
				{
					Text:      fmt.Sprintf("Salient telemetry for %s", c1),
					Salience:  0.90,
					Timestamp: eventTime,
				},
			}
		} else {
			// Transient noise trace: one-off ephemeral terms
			noiseID := rng.Intn(50000)
			trace.TaskGoal = fmt.Sprintf("Ephemeral speculative routine %d", noiseID)
			if rng.Float64() < 0.6 {
				trace.Outcome = model.OutcomeFailure
			} else {
				trace.Outcome = model.OutcomeNeutral
			}
			trace.Trajectory = []model.TrajectoryStep{
				{
					StepIndex:   1,
					Thought:     fmt.Sprintf("Investigating transient sensor jitter code_%d", noiseID),
					Action:      fmt.Sprintf("probe_jitter_%d", noiseID),
					Observation: "Transient condition cleared",
					Status:      "neutral",
					Timestamp:   eventTime,
				},
			}
		}

		if err := sqliteStore.QueueTrace(ctx, trace); err != nil {
			fmt.Printf("[FATAL] Error queueing trace %d: %v\n", i+1, err)
			os.Exit(1)
		}
	}

	ingestDuration := time.Since(ingestStart)
	fmt.Printf("         Queued %d traces in %.2fs (%.1f traces/sec)\n",
		*eventCount, ingestDuration.Seconds(), float64(*eventCount)/ingestDuration.Seconds())

	// 2. Execute intermediate consolidation cycles across simulated timeline
	fmt.Println("[Step 2] Executing background consolidation and graph fusion cycle...")
	fuseStart := time.Now()
	fuseResp, err := engine.RunConsolidationCycle(ctx, baseTime.Add(7*24*time.Hour))
	if err != nil {
		fmt.Printf("[FATAL] Initial consolidation cycle failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("         Fused %d traces into knowledge graph in %.2fms\n",
		fuseResp.TracesFused, time.Since(fuseStart).Seconds()*1000.0)

	initialStats, _ := engine.GetStats(ctx)
	fmt.Printf("[Mid-Cycle Topology] Active Nodes: %d | Active Edges: %d | Mean Stability: %.3f\n",
		initialStats.ActiveNodes, initialStats.ActiveEdges, initialStats.MeanStabilityScore)

	// 3. Advance time past grace period to evaluate exponential decay and soft-archival
	fmt.Println("[Step 3] Advancing simulation clock by 14 days to evaluate decay & pruning...")
	evalTime := baseTime.Add(21 * 24 * time.Hour) // 21 days total (> 7-day grace period and multiple half-lives)

	decayStart := time.Now()
	cycleResp, err := engine.RunConsolidationCycle(ctx, evalTime)
	if err != nil {
		fmt.Printf("[FATAL] Final decay cycle failed: %v\n", err)
		os.Exit(1)
	}
	decayDuration := time.Since(decayStart)

	finalStats, err := engine.GetStats(ctx)
	if err != nil {
		fmt.Printf("[FATAL] Failed fetching final stats: %v\n", err)
		os.Exit(1)
	}

	// 4. Verify hub stabilization vs transient noise decay
	fmt.Println("[Step 4] Verifying hub stabilization and noise elimination criteria...")

	var stableHubs int
	hubScores := make([]float64, 0, len(hubConcepts))

	for _, hub := range hubConcepts {
		node, err := sqliteStore.FindMatchingNode(ctx, hub, "concept")
		if err == nil && node != nil {
			hubScores = append(hubScores, node.StabilityScore)
			if node.StabilityScore >= 0.70 && !node.IsArchived {
				stableHubs++
			}
		}
	}

	// Query soft-archived and active nodes count from final stats
	archivedCount := finalStats.ArchivedNodes
	activeCount := finalStats.ActiveNodes

	sort.Float64s(hubScores)
	meanHubScore := 0.0
	for _, s := range hubScores {
		meanHubScore += s
	}
	if len(hubScores) > 0 {
		meanHubScore /= float64(len(hubScores))
	}

	// 5. Output Verification Report
	fmt.Println("-----------------------------------------------------------------")
	fmt.Println("                 EMPIRICAL VALIDATION RESULTS                    ")
	fmt.Println("-----------------------------------------------------------------")
	fmt.Printf(" Total Episodic Events:    %d\n", *eventCount)
	fmt.Printf(" Ingestion Throughput:     %.1f traces/sec\n", float64(*eventCount)/ingestDuration.Seconds())
	fmt.Printf(" Cycle Execution Duration: %.2f ms\n", float64(decayDuration.Microseconds())/1000.0)
	fmt.Printf(" Active Nodes (Retained):  %d\n", activeCount)
	fmt.Printf(" Soft-Archived Nodes:      %d (Transient noise pruned)\n", archivedCount)
	fmt.Printf(" Active Relational Edges:  %d\n", finalStats.ActiveEdges)
	fmt.Printf(" Pruned Relational Edges:  %d\n", cycleResp.EdgesPruned)
	fmt.Printf(" Stable Core Hubs Found:   %d / %d (Mean Stability: %.3f)\n", stableHubs, len(hubConcepts), meanHubScore)
	fmt.Printf(" Mean Active Stability:    %.3f\n", finalStats.MeanStabilityScore)
	fmt.Printf(" Mean Active Edge Weight:  %.3f\n", finalStats.MeanEdgeWeight)
	fmt.Println("-----------------------------------------------------------------")

	// Assertion checks
	passNoisePruning := archivedCount > 0 && float64(archivedCount)/float64(*eventCount) >= 0.40
	passHubStabilization := stableHubs >= len(hubConcepts)/2 && meanHubScore >= 0.65
	passBoundedGrowth := activeCount < int64(*eventCount)

	if passNoisePruning && passHubStabilization && passBoundedGrowth {
		fmt.Println(" RESULT: PASS - 1,000 synthetic episodic task events validation")
		fmt.Println(" successfully proved key hub entity stabilization and clean decay")
		fmt.Println(" of transient noise, preventing unbounded SQLite bloat!")
		fmt.Println("=================================================================")
	} else {
		fmt.Printf(" RESULT: FAIL - Verification criteria not satisfied (NoisePruned: %v, HubsStable: %v, Bounded: %v)\n",
			passNoisePruning, passHubStabilization, passBoundedGrowth)
		fmt.Println("=================================================================")
		os.Exit(1)
	}
}
