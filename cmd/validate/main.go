package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/embedding"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/recall"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

func main() {
	nodeCount := flag.Int("nodes", 10000, "Number of synthetic knowledge nodes to generate")
	edgeCount := flag.Int("edges", 25000, "Number of synthetic relational edges to generate")
	dimensions := flag.Int("dims", model.DefaultVectorDim, "Embedding vector dimensionality (default 384-D)")
	queries := flag.Int("queries", 100, "Number of test recall queries to execute")
	dbPath := flag.String("db", "test_benchmark.db", "Temporary SQLite database path for benchmark")
	flag.Parse()

	fmt.Println("================================================================")
	fmt.Println("   Sekha Long-Term Knowledge Graph Store Benchmark Suite        ")
	fmt.Println("   Node 1 (8GB RAM Pi 5) Associative Recall & Vector Benchmark ")
	fmt.Println("================================================================")
	fmt.Printf(" Target Nodes:      %d\n", *nodeCount)
	fmt.Printf(" Target Edges:      %d\n", *edgeCount)
	fmt.Printf(" Vector Dims:       %d-D float32\n", *dimensions)
	fmt.Printf(" Benchmark Queries: %d iterations\n", *queries)
	fmt.Printf(" DB File:           %s\n", *dbPath)
	fmt.Println("----------------------------------------------------------------")

	// Remove any leftover temporary test db
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

	ctx := context.Background()

	// 1. Synthetic Data Generation
	fmt.Printf("[Step 1] Generating and ingesting %d synthetic nodes...\n", *nodeCount)
	genStart := time.Now()

	rng := rand.New(rand.NewSource(42))
	entityTypes := []string{"concept", "episodic_event", "decision", "procedure", "telemetry_rule"}
	relationTypes := []string{"relates_to", "caused_by", "subgoal_of", "associates_with", "precedes"}

	const batchSize = 1000
	nodes := make([]model.Node, 0, *nodeCount)
	now := time.Now().UTC()

	for i := 0; i < *nodeCount; i++ {
		vec := make([]float32, *dimensions)
		for d := 0; d < *dimensions; d++ {
			vec[d] = float32(rng.NormFloat64())
		}
		var sum float64
		for _, v := range vec {
			sum += float64(v * v)
		}
		mag := float32(math.Sqrt(sum))
		if mag > 0 {
			for d := 0; d < *dimensions; d++ {
				vec[d] /= mag
			}
		}

		elapsedHours := rng.Float64() * 720.0
		lastAccessed := now.Add(-time.Duration(elapsedHours * float64(time.Hour)))
		created := lastAccessed.Add(-time.Duration(rng.Float64() * 100.0 * float64(time.Hour)))
		accessCount := int64(rng.Intn(150))

		nodes = append(nodes, model.Node{
			ID:             fmt.Sprintf("node-%06d", i+1),
			EntityType:     entityTypes[rng.Intn(len(entityTypes))],
			Label:          fmt.Sprintf("Synthetic Knowledge Entity #%d", i+1),
			Summary:        fmt.Sprintf("Episodic context and semantic knowledge summary for entity %d.", i+1),
			Embedding:      vec,
			CreatedAt:      created,
			LastAccessedAt: lastAccessed,
			AccessCount:    accessCount,
			StabilityScore: 0.5 + rng.Float64()*0.5,
		})

		if len(nodes) >= batchSize || i == *nodeCount-1 {
			if _, err := sqliteStore.InsertNodes(ctx, nodes); err != nil {
				fmt.Printf("[FATAL] Error inserting node batch: %v\n", err)
				os.Exit(1)
			}
			nodes = nodes[:0]
		}
	}
	fmt.Printf("         Inserted %d nodes in %.2fs\n", *nodeCount, time.Since(genStart).Seconds())

	// 2. Generate and ingest 25,000 edges
	fmt.Printf("[Step 2] Generating and ingesting %d relational edges...\n", *edgeCount)
	edgeStart := time.Now()
	edges := make([]model.Edge, 0, *edgeCount)

	for i := 0; i < *edgeCount; i++ {
		srcIdx := rng.Intn(*nodeCount) + 1
		tgtIdx := rng.Intn(*nodeCount) + 1
		for tgtIdx == srcIdx {
			tgtIdx = rng.Intn(*nodeCount) + 1
		}

		edges = append(edges, model.Edge{
			SourceID:     fmt.Sprintf("node-%06d", srcIdx),
			TargetID:     fmt.Sprintf("node-%06d", tgtIdx),
			RelationType: relationTypes[rng.Intn(len(relationTypes))],
			Weight:       0.2 + rng.Float64()*0.8,
			CreatedAt:    now.Add(-time.Duration(rng.Float64() * 500 * float64(time.Hour))),
		})

		if len(edges) >= batchSize || i == *edgeCount-1 {
			if _, err := sqliteStore.InsertEdges(ctx, edges); err != nil {
				fmt.Printf("[FATAL] Error inserting edge batch: %v\n", err)
				os.Exit(1)
			}
			edges = edges[:0]
		}
	}
	fmt.Printf("         Inserted %d edges in %.2fs\n", *edgeCount, time.Since(edgeStart).Seconds())

	// 3. Hydrate in-memory associative recall engine
	fmt.Println("[Step 3] Initialising and hydrating in-memory associative recall engine...")
	hydrateStart := time.Now()
	engineCfg := recall.DefaultConfig()
	engineCfg.Embedder = embedding.NewMockEmbedder(*dimensions)
	engine, err := recall.NewEngine(ctx, sqliteStore, engineCfg)
	if err != nil {
		fmt.Printf("[FATAL] Failed to initialise recall engine: %v\n", err)
		os.Exit(1)
	}
	hydrateMS := float64(time.Since(hydrateStart).Microseconds()) / 1000.0
	fmt.Printf("         Hydrated index for %d nodes in %.2f ms (Target: < 150.0 ms)\n", *nodeCount, hydrateMS)
	if hydrateMS >= 150.0 {
		fmt.Printf("[FATAL] Hydration latency exceeded 150 ms threshold: %.2f ms\n", hydrateMS)
		os.Exit(1)
	}

	// 4. Graph Summary Check
	summary, err := sqliteStore.GetGraphSummary(ctx)
	if err != nil {
		fmt.Printf("[FATAL] Failed getting summary: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("[Topology] Node Count: %d | Edge Count: %d | Avg Degree: %.2f | Density: %.6f\n",
		summary.NodeCount, summary.EdgeCount, summary.AvgDegree, summary.GraphDensity)

	// 5. Benchmark Associative Recall Queries
	fmt.Printf("[Step 4] Executing %d hybrid associative recall queries (Sim + Frequency + Recency + 1-Hop)...\n", *queries)

	latencies := make([]float64, 0, *queries)
	var totalLatency float64

	for q := 0; q < *queries; q++ {
		qVec := make([]float32, *dimensions)
		for d := 0; d < *dimensions; d++ {
			qVec[d] = float32(rng.NormFloat64())
		}
		var sum float64
		for _, v := range qVec {
			sum += float64(v * v)
		}
		mag := float32(math.Sqrt(sum))
		for d := 0; d < *dimensions; d++ {
			qVec[d] /= mag
		}

		req := model.RecallRequest{
			Embedding:  qVec,
			TopK:       10,
			Alpha:      0.6,
			Beta:       0.2,
			Gamma:      0.2,
			ExpandHops: 1,
		}

		resp, err := engine.Recall(ctx, req)
		if err != nil {
			fmt.Printf("[FATAL] Recall failed at query %d: %v\n", q, err)
			os.Exit(1)
		}

		if len(resp.Nodes) == 0 {
			fmt.Printf("[FATAL] Expected returned nodes at query %d, got 0\n", q)
			os.Exit(1)
		}

		latencies = append(latencies, resp.QueryLatencyMS)
		totalLatency += resp.QueryLatencyMS
	}

	sort.Float64s(latencies)

	mean := totalLatency / float64(len(latencies))
	p50 := latencies[len(latencies)*50/100]
	p90 := latencies[len(latencies)*90/100]
	p95 := latencies[len(latencies)*95/100]
	p99 := latencies[len(latencies)*99/100]
	maxLat := latencies[len(latencies)-1]
	minLat := latencies[0]

	targetLatency := 15.0 // Target is < 15.0 ms per acceptance criteria

	fmt.Println("----------------------------------------------------------------")
	fmt.Println("                    EMPIRICAL BENCHMARK RESULTS                 ")
	fmt.Println("----------------------------------------------------------------")
	fmt.Printf(" Total Synthetic Nodes: %d\n", *nodeCount)
	fmt.Printf(" Total Relational Edges: %d\n", *edgeCount)
	fmt.Printf(" Hydration Latency:     %.3f ms (Target: < 150.0 ms)\n", hydrateMS)
	fmt.Printf(" Mean Recall Latency:   %.3f ms (Target: < %.1f ms)\n", mean, targetLatency)
	fmt.Printf(" Median (p50) Latency:  %.3f ms\n", p50)
	fmt.Printf(" 90th Percentile (p90): %.3f ms\n", p90)
	fmt.Printf(" 95th Percentile (p95): %.3f ms (Target: < %.1f ms)\n", p95, targetLatency)
	fmt.Printf(" 99th Percentile (p99): %.3f ms\n", p99)
	fmt.Printf(" Min / Max Latency:     %.3f ms / %.3f ms\n", minLat, maxLat)
	fmt.Println("----------------------------------------------------------------")

	if p95 < targetLatency && mean < targetLatency {
		fmt.Printf(" RESULT: PASS - Associative recall latency on %d nodes\n", *nodeCount)
		fmt.Printf(" satisfies the sub-15ms requirement by a factor of %.1fx!\n", targetLatency/mean)
	} else {
		fmt.Printf(" RESULT: FAIL - Recall latency exceeded %.1fms threshold\n", targetLatency)
		os.Exit(1)
	}

	time.Sleep(50 * time.Millisecond)

	// 6. Multi-Project Anchor Recall Benchmark
	fmt.Println("\n[Step 5] Multi-Project Anchor Benchmark (Eliminating Generic Token Dilution)...")
	projects := []struct {
		name   string
		anchor string
	}{
		{name: "Kestrel", anchor: "#project:kestrel"},
		{name: "Falcon", anchor: "#project:falcon"},
		{name: "Merlin", anchor: "#project:merlin"},
	}

	genericTopics := []string{
		"network port timeout settings",
		"buffer retry queue overflow threshold",
		"keepalive heartbeat socket drop",
		"rpc connection deadline limit",
		"circuit breaker error rate tripping",
	}

	var anchorNodes []model.Node
	for pIdx, proj := range projects {
		for tIdx, topic := range genericTopics {
			anchorNodes = append(anchorNodes, model.Node{
				ID:             fmt.Sprintf("node-mp-%d-%d", pIdx+1, tIdx+1),
				EntityType:     "concept",
				Label:          fmt.Sprintf("%s %s", proj.name, topic),
				Summary:        fmt.Sprintf("Project %s operational configuration for %s", proj.name, topic),
				Anchors:        []string{proj.anchor},
				CreatedAt:      now.Add(-time.Duration(pIdx*24) * time.Hour),
				LastAccessedAt: now.Add(-time.Duration(pIdx*12) * time.Hour),
				AccessCount:    int64(10 + (3-pIdx)*20),
				StabilityScore: 1.0,
			})
		}
	}

	if _, err := sqliteStore.InsertNodes(ctx, anchorNodes); err != nil {
		fmt.Printf("[FATAL] Error inserting anchor benchmark nodes: %v\n", err)
		os.Exit(1)
	}
	engine.RegisterNodes(anchorNodes)

	anchorQueries := 100
	anchorHits := 0
	anchorLatencies := make([]float64, 0, anchorQueries)
	var anchorTotalLatency float64

	for i := 0; i < anchorQueries; i++ {
		targetProj := projects[rng.Intn(len(projects))]
		topic := genericTopics[rng.Intn(len(genericTopics))]

		req := model.RecallRequest{
			Query:      topic,
			Anchors:    []string{targetProj.anchor},
			AnchorMode: "boost",
			TopK:       5,
		}

		resp, err := engine.Recall(ctx, req)
		if err != nil {
			fmt.Printf("[FATAL] Anchor query %d failed: %v\n", i, err)
			os.Exit(1)
		}

		if len(resp.Nodes) == 0 {
			fmt.Printf("[FATAL] Anchor query %d returned 0 nodes\n", i)
			os.Exit(1)
		}

		anchorLatencies = append(anchorLatencies, resp.QueryLatencyMS)
		anchorTotalLatency += resp.QueryLatencyMS

		for _, a := range resp.Nodes[0].Anchors {
			if a == targetProj.anchor {
				anchorHits++
				break
			}
		}
	}

	sort.Float64s(anchorLatencies)
	anchorMean := anchorTotalLatency / float64(len(anchorLatencies))
	anchorP95 := anchorLatencies[len(anchorLatencies)*95/100]
	hitRate := float64(anchorHits) / float64(anchorQueries) * 100.0

	fmt.Println("----------------------------------------------------------------")
	fmt.Println("             MULTI-PROJECT ANCHOR BENCHMARK RESULTS             ")
	fmt.Println("----------------------------------------------------------------")
	fmt.Printf(" Multi-Project Nodes:   %d (3 projects, overlapping generic tokens)\n", len(anchorNodes))
	fmt.Printf(" Benchmark Queries:     %d iterations\n", anchorQueries)
	fmt.Printf(" Target Rank 1 Hit:     %.1f%% (Target: >= 90.0%%)\n", hitRate)
	fmt.Printf(" Mean Query Latency:    %.3f ms (Target: < 15.0 ms)\n", anchorMean)
	fmt.Printf(" 95th Percentile (p95): %.3f ms (Target: < 15.0 ms)\n", anchorP95)
	fmt.Println("----------------------------------------------------------------")

	if hitRate >= 90.0 && anchorP95 < targetLatency {
		fmt.Println(" RESULT: PASS - Multi-project anchor tags eliminate ranking dilution")
		fmt.Println(" with >= 90% Rank 1 precision and sub-15ms edge query latency!")
	} else {
		fmt.Printf(" RESULT: FAIL - Anchor benchmark did not satisfy requirements (Hit Rate: %.1f%%, p95: %.3fms)\n", hitRate, anchorP95)
		os.Exit(1)
	}

	// 7. Acceptance Criteria Verification
	fmt.Println("\n[Step 6] Verifying Acceptance Criteria (Semantic Paraphrasing, Lexical Fidelity, Noise Floor)...")

	// Ingest acceptance criteria nodes
	acNodes := []model.Node{
		{
			ID:             "ac-node-consensus",
			EntityType:     "decision",
			Label:          "Cluster Consensus Settings",
			Summary:        "consensus heartbeat timeout configured to 500ms for health checks",
			CreatedAt:      now,
			LastAccessedAt: now,
			AccessCount:    20,
			StabilityScore: 1.0,
		},
		{
			ID:             "ac-node-nginx",
			EntityType:     "procedure",
			Label:          "Web Server Concurrency",
			Summary:        "Set worker_connections 4096 in nginx core events configuration",
			CreatedAt:      now,
			LastAccessedAt: now,
			AccessCount:    20,
			StabilityScore: 1.0,
		},
		{
			ID:             "ac-node-port",
			EntityType:     "decision",
			Label:          "Memory Service Daemon",
			Summary:        "Sekha knowledge graph API listening on port 8084",
			CreatedAt:      now,
			LastAccessedAt: now,
			AccessCount:    20,
			StabilityScore: 1.0,
		},
	}

	if _, err := sqliteStore.InsertNodes(ctx, acNodes); err != nil {
		fmt.Printf("[FATAL] Error inserting acceptance criteria nodes: %v\n", err)
		os.Exit(1)
	}
	engine.RegisterNodes(acNodes)

	// Verification 1: Semantic Paraphrasing
	paraResp, err := engine.Recall(ctx, model.RecallRequest{
		Query: "cluster coordination interval",
		TopK:  3,
	})
	if err != nil || len(paraResp.Nodes) == 0 {
		fmt.Printf("[FATAL] Semantic paraphrase recall query failed: %v\n", err)
		os.Exit(1)
	}
	paraRank1 := paraResp.Nodes[0]
	var paraDense float64
	if paraRank1.DenseScore != nil {
		paraDense = *paraRank1.DenseScore
	}
	fmt.Printf(" [Criterion 1] Semantic Paraphrase: Rank 1 = %s (S_dense = %.4f, S_hybrid = %.4f)\n",
		paraRank1.ID, paraDense, paraRank1.SimScore)
	if paraRank1.ID != "ac-node-consensus" || paraDense <= 0.70 {
		fmt.Printf("[FATAL] Criterion 1 failed: expected ac-node-consensus with S_dense > 0.70\n")
		os.Exit(1)
	}

	// Verification 2: Lexical Keyword Fidelity
	lexResp, err := engine.Recall(ctx, model.RecallRequest{
		Query: "worker_connections 4096 reverse proxy tuning",
		TopK:  3,
	})
	if err != nil || len(lexResp.Nodes) == 0 {
		fmt.Printf("[FATAL] Lexical fidelity recall query failed: %v\n", err)
		os.Exit(1)
	}
	lexRank1 := lexResp.Nodes[0]
	var lexBM25 float64
	if lexRank1.BM25Score != nil {
		lexBM25 = *lexRank1.BM25Score
	}
	fmt.Printf(" [Criterion 2] Lexical Keyword Fidelity: Rank 1 = %s (S_bm25 = %.4f, S_hybrid = %.4f)\n",
		lexRank1.ID, lexBM25, lexRank1.SimScore)
	if lexRank1.ID != "ac-node-nginx" {
		fmt.Printf("[FATAL] Criterion 2 failed: expected ac-node-nginx at rank 1\n")
		os.Exit(1)
	}

	// Verification 3: Noise Floor Suppression
	noiseResp, err := engine.Recall(ctx, model.RecallRequest{
		Query: "Petrelwick deployment",
		TopK:  3,
	})
	if err != nil || len(noiseResp.Nodes) == 0 {
		fmt.Printf("[FATAL] Noise suppression query failed: %v\n", err)
		os.Exit(1)
	}
	noiseTop := noiseResp.Nodes[0]
	fmt.Printf(" [Criterion 3] Noise Suppression: Top Similarity = %.4f (Target: < 0.10)\n", noiseTop.SimScore)
	if noiseTop.SimScore >= 0.10 {
		fmt.Printf("[FATAL] Criterion 3 failed: expected sim < 0.10 for out-of-vocabulary query, got %.4f\n", noiseTop.SimScore)
		os.Exit(1)
	}

	fmt.Println("----------------------------------------------------------------")
	fmt.Println(" RESULT: PASS - All 5 Acceptance & Verification Criteria Passed!")
	fmt.Println("================================================================")
}
