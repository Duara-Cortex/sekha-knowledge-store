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
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/recall"
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
		isHub := (i%5 == 0) // 20% recurrent hub
		var eventTime time.Time
		if isHub {
			// Recurrent hub entities occur across the full operational period (up to Day 14)
			eventTime = baseTime.Add(time.Duration(rng.Float64()*14.0*24.0) * time.Hour)
		} else {
			// Transient noise events are strictly concentrated in early days (Days 0 to 5)
			// ensuring elapsed inactivity >= 7-day grace period at Day 14
			eventTime = baseTime.Add(time.Duration(rng.Float64()*5.0*24.0) * time.Hour)
		}

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

	// 2. Execute intermediate consolidation cycles to fuse all queued traces
	fmt.Println("[Step 2] Executing background consolidation and graph fusion cycles...")
	fuseStart := time.Now()
	totalTracesFused := 0
	for {
		fuseResp, err := engine.RunConsolidationCycle(ctx, baseTime.Add(7*24*time.Hour))
		if err != nil {
			fmt.Printf("[FATAL] Consolidation cycle failed: %v\n", err)
			os.Exit(1)
		}
		totalTracesFused += fuseResp.TracesFused
		if fuseResp.TracesFused == 0 {
			break
		}
	}
	fmt.Printf("         Fused %d traces into knowledge graph in %.2fms\n",
		totalTracesFused, time.Since(fuseStart).Seconds()*1000.0)

	initialStats, _ := engine.GetStats(ctx)
	peakActiveNodes := initialStats.ActiveNodes
	fmt.Printf("[Mid-Cycle Topology] Active Nodes (Peak): %d | Active Edges: %d | Mean Stability: %.3f\n",
		peakActiveNodes, initialStats.ActiveEdges, initialStats.MeanStabilityScore)

	// 3. Advance time to Day 14 to evaluate exponential decay and soft-archival
	fmt.Println("[Step 3] Advancing simulation clock to Day 14 to evaluate decay & pruning...")
	evalTime := baseTime.Add(14 * 24 * time.Hour) // Day 14: transient noise (Days 0-5) is 9-14 days old (> 7d grace)

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
	fmt.Printf(" Peak Active Nodes:        %d\n", peakActiveNodes)
	fmt.Printf(" Active Nodes (Retained):  %d\n", activeCount)
	fmt.Printf(" Soft-Archived Nodes:      %d (Transient noise pruned)\n", archivedCount)
	fmt.Printf(" Active Relational Edges:  %d\n", finalStats.ActiveEdges)
	fmt.Printf(" Pruned Relational Edges:  %d\n", cycleResp.EdgesPruned)
	fmt.Printf(" Stable Core Hubs Found:   %d / %d (Mean Stability: %.3f)\n", stableHubs, len(hubConcepts), meanHubScore)
	fmt.Printf(" Mean Active Stability:    %.3f\n", finalStats.MeanStabilityScore)
	fmt.Printf(" Mean Active Edge Weight:  %.3f\n", finalStats.MeanEdgeWeight)
	fmt.Println("-----------------------------------------------------------------")

	// Assertion checks:
	// 1. Noise Pruning: transient noise was pruned and archived
	passNoisePruning := archivedCount > 0 && float64(archivedCount)/float64(peakActiveNodes) >= 0.40
	// 2. Hub Stabilization: core recurrent concepts stabilized with high scores
	passHubStabilization := stableHubs >= len(hubConcepts)/2 && meanHubScore >= 0.65
	// 3. Bounded Growth: post-decay active nodes are bounded well below peak expansion
	passBoundedGrowth := activeCount < peakActiveNodes && activeCount <= peakActiveNodes/2

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

	// =================================================================
	// Phase 2: Accelerated Decay & Intrinsic Importance Verification
	// =================================================================
	fmt.Println("\n=================================================================")
	fmt.Println("   Phase 2: Accelerated Decay & Intrinsic Importance Audit       ")
	fmt.Println("=================================================================")

	// 1. Ingest 100 Core Anchored / High-Importance Knowledge Nodes
	fmt.Println("[Step 1] Ingesting 100 core anchored & high-importance configuration hubs...")
	coreNodes := make([]model.Node, 0, 100)
	now := time.Now().UTC()
	for i := 0; i < 50; i++ {
		coreNodes = append(coreNodes, model.Node{
			ID:              fmt.Sprintf("core-anchored-%03d", i+1),
			EntityType:      "concept",
			Label:           fmt.Sprintf("Core Architecture Anchor %d", i+1),
			Summary:         fmt.Sprintf("Critical project architecture definition for subsystem %d", i+1),
			Anchors:         []string{fmt.Sprintf("#project:kestrel-%d", i%5)},
			ImportanceScore: 0.85,
			StabilityScore:  1.0,
			CreatedAt:       now,
			LastAccessedAt:  now,
		})
	}
	for i := 0; i < 50; i++ {
		coreNodes = append(coreNodes, model.Node{
			ID:              fmt.Sprintf("core-config-%03d", i+1),
			EntityType:      "decision",
			Label:           fmt.Sprintf("Cluster Master System Config %d", i+1),
			Summary:         fmt.Sprintf("INGEST_PORT and API_KEY settings for node cluster %d", i+1),
			ImportanceScore: 0.90, // Unanchored but protected by importance >= 0.80
			StabilityScore:  1.0,
			CreatedAt:       now,
			LastAccessedAt:  now,
		})
	}
	if _, err := sqliteStore.InsertNodes(ctx, coreNodes); err != nil {
		fmt.Printf("[FATAL] Failed inserting core knowledge nodes: %v\n", err)
		os.Exit(1)
	}

	// 2. Ingest 1,000 Distractor Nodes
	fmt.Println("[Step 2] Ingesting 1,000 unanchored distractor / transient telemetry events...")
	const distractorCount = 1000
	distractors := make([]model.Node, 0, distractorCount)
	for i := 0; i < distractorCount; i++ {
		distractors = append(distractors, model.Node{
			ID:              fmt.Sprintf("distractor-%04d", i+1),
			EntityType:      "telemetry",
			Label:           fmt.Sprintf("Transient telemetry jitter code_%d", i+1),
			Summary:         fmt.Sprintf("Ephemeral telemetry reading and sensor jitter sample %d", i+1),
			ImportanceScore: 0.25 + rng.Float64()*0.10, // 0.25 - 0.35 (<= 0.40)
			StabilityScore:  0.30 + rng.Float64()*0.20,
			CreatedAt:       now.Add(-2 * time.Hour),
			LastAccessedAt:  now.Add(-2 * time.Hour),
		})
	}
	if _, err := sqliteStore.InsertNodes(ctx, distractors); err != nil {
		fmt.Printf("[FATAL] Failed inserting distractor nodes: %v\n", err)
		os.Exit(1)
	}

	// 3. Trigger Scoped Accelerated Decay Cycle
	fmt.Println("[Step 3] Executing on-demand accelerated decay cycle (scope: unanchored, grace: 0s)...")
	zeroGrace := 0 * time.Second
	oneHour := 1 * time.Hour
	pruneThresh := 0.50

	accDecayStart := time.Now()
	_, err = engine.ExecuteScopedDecay(ctx, model.DecayRequest{
		Scope:                 "unanchored",
		InactivityGracePeriod: &zeroGrace,
		DecayHalfLife:         &oneHour,
		PruneThreshold:        &pruneThresh,
		MinImportanceToRetain: 0.80,
	}, now)
	if err != nil {
		fmt.Printf("[FATAL] Scoped accelerated decay failed: %v\n", err)
		os.Exit(1)
	}
	accDecayDuration := time.Since(accDecayStart)
	accDecayDurationMS := float64(accDecayDuration.Microseconds()) / 1000.0

	// 4. Verify 100% of 1,000 distractor nodes are soft-archived, 100% of core hubs remain active
	fmt.Println("[Step 4] Verifying 100% distractor archival and core hub immunity...")
	archivedDistractors := 0
	for i := 0; i < distractorCount; i++ {
		n, err := sqliteStore.GetNode(ctx, fmt.Sprintf("distractor-%04d", i+1))
		if err == nil && n != nil && n.IsArchived {
			archivedDistractors++
		}
	}

	activeCoreNodes := 0
	for _, cn := range coreNodes {
		n, err := sqliteStore.GetNode(ctx, cn.ID)
		if err == nil && n != nil && !n.IsArchived {
			activeCoreNodes++
		}
	}

	// 5. Verify Frequency Starvation Protection in Associative Recall
	fmt.Println("[Step 5] Auditing associative recall frequency starvation protection...")
	recallCfg := recall.DefaultConfig()
	recallEngine, err := recall.NewEngine(ctx, sqliteStore, recallCfg)
	if err != nil {
		fmt.Printf("[FATAL] Failed initialising recall engine: %v\n", err)
		os.Exit(1)
	}

	// Node A: Critical system config, access_count=1, importance=0.90
	// Node B: High-frequency distractor, access_count=50, importance=0.30
	nodeA := model.Node{
		ID:              "starvation-test-config",
		EntityType:      "concept",
		Label:           "Cluster Master Config INGEST_PORT",
		Summary:         "INGEST_PORT binding and cluster network bounds",
		ImportanceScore: 0.90,
		AccessCount:     1,
		StabilityScore:  1.0,
		CreatedAt:       now,
		LastAccessedAt:  now,
	}
	nodeB := model.Node{
		ID:              "starvation-test-distractor",
		EntityType:      "telemetry",
		Label:           "Cluster Master Telemetry Jitter",
		Summary:         "High frequency telemetry logs for cluster master",
		ImportanceScore: 0.30,
		AccessCount:     50,
		StabilityScore:  1.0,
		CreatedAt:       now,
		LastAccessedAt:  now,
	}
	_, _ = sqliteStore.InsertNodes(ctx, []model.Node{nodeA, nodeB})
	recallEngine.RegisterNodes([]model.Node{nodeA, nodeB})

	recallStart := time.Now()
	recallResp, err := recallEngine.Recall(ctx, model.RecallRequest{
		Query: "Cluster Master Config",
		TopK:  2,
	})
	recallDuration := time.Since(recallStart)
	recallDurationMS := float64(recallDuration.Microseconds()) / 1000.0

	if err != nil {
		fmt.Printf("[FATAL] Recall query failed: %v\n", err)
		os.Exit(1)
	}

	starvationProtected := len(recallResp.Nodes) >= 2 && recallResp.Nodes[0].ID == "starvation-test-config"

	// 6. Empirical Validation Results Report
	fmt.Println("-----------------------------------------------------------------")
	fmt.Println("      ACCELERATED DECAY & IMPORTANCE AUDIT RESULTS               ")
	fmt.Println("-----------------------------------------------------------------")
	fmt.Printf(" Distractor Nodes Ingested: %d\n", distractorCount)
	fmt.Printf(" Distractor Nodes Archived: %d (%.1f%%, Target: 100.0%%)\n",
		archivedDistractors, float64(archivedDistractors)/float64(distractorCount)*100.0)
	fmt.Printf(" Core Hub Nodes Retained:   %d / %d (%.1f%%, Target: 100.0%%)\n",
		activeCoreNodes, len(coreNodes), float64(activeCoreNodes)/float64(len(coreNodes))*100.0)
	fmt.Printf(" Scoped Decay Execution:    %.2f ms (Target: < 100.0 ms)\n", accDecayDurationMS)
	fmt.Printf(" Associative Recall Latency:%.2f ms (Target: < 15.0 ms)\n", recallDurationMS)
	if len(recallResp.Nodes) >= 2 {
		fmt.Printf(" Rank 1 (Config):           %s (Score: %.4f, Imp: %.2f, Access: %d)\n",
			recallResp.Nodes[0].ID, recallResp.Nodes[0].Score, recallResp.Nodes[0].ImportanceScore, recallResp.Nodes[0].AccessCount)
		fmt.Printf(" Rank 2 (Distractor):       %s (Score: %.4f, Imp: %.2f, Access: %d)\n",
			recallResp.Nodes[1].ID, recallResp.Nodes[1].Score, recallResp.Nodes[1].ImportanceScore, recallResp.Nodes[1].AccessCount)
	}
	fmt.Printf(" Frequency Starvation Saved: %v\n", starvationProtected)
	fmt.Println("-----------------------------------------------------------------")

	passDistractorPrune := archivedDistractors == distractorCount
	passHubImmunity := activeCoreNodes == len(coreNodes)
	passDecayLatency := accDecayDurationMS < 100.0
	passRecallLatency := recallDurationMS < 15.0

	if passDistractorPrune && passHubImmunity && passDecayLatency && passRecallLatency && starvationProtected {
		fmt.Println(" RESULT: PASS - Accelerated decay and intrinsic importance scoring")
		fmt.Println(" verified! Critical configs protected and 100% distractors pruned.")
		fmt.Println("=================================================================")
	} else {
		fmt.Printf(" RESULT: FAIL - Verification targets not met (Prune: %v, Hubs: %v, DecayLat: %v, RecallLat: %v, Starvation: %v)\n",
			passDistractorPrune, passHubImmunity, passDecayLatency, passRecallLatency, starvationProtected)
		fmt.Println("=================================================================")
		os.Exit(1)
	}
}
