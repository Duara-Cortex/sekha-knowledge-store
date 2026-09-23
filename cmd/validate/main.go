package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
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

	// Step 7: Dedicated Gated 1-Hop Graph Expansion & Displacement Prevention Benchmark
	runGatedExpansionBenchmark(ctx)

	fmt.Println("\n================================================================")
	fmt.Println(" FINAL RESULT: PASS - All Tasks & Benchmarks Successfully Passed!")
	fmt.Println("================================================================")
}

// runGatedExpansionBenchmark executes the dedicated 216-node / 866-edge and scaled 1,000-node / 4,000-edge benchmarks
// validating gated 1-hop graph expansion, zero distractor displacement, and sub-5ms query latency.
func runGatedExpansionBenchmark(ctx context.Context) {
	fmt.Println("\n================================================================")
	fmt.Println("[Step 7] Dedicated Gated 1-Hop Graph Expansion Benchmark        ")
	fmt.Println("         (216-Node/866-Edge & 1,000-Node/4,000-Edge Graphs)     ")
	fmt.Println("================================================================")

	// Part A: 216 Nodes / 866 Edges Graph
	dbPath216 := "test_benchmark_gated_216.db"
	_ = os.Remove(dbPath216)
	_ = os.Remove(dbPath216 + "-wal")
	_ = os.Remove(dbPath216 + "-shm")
	defer func() {
		_ = os.Remove(dbPath216)
		_ = os.Remove(dbPath216 + "-wal")
		_ = os.Remove(dbPath216 + "-shm")
	}()

	store216, err := store.NewSQLiteStore(dbPath216)
	if err != nil {
		fmt.Printf("[FATAL] Failed to initialise 216-node store: %v\n", err)
		os.Exit(1)
	}
	defer store216.Close()

	rng := rand.New(rand.NewSource(42))
	now := time.Now().UTC()

	topics := []string{
		"network port timeout settings",
		"buffer retry queue overflow threshold",
		"keepalive heartbeat socket drop",
		"rpc connection deadline limit",
		"circuit breaker error rate tripping",
		"ingress gateway listen port binding",
		"memory buffer pool eviction policy",
		"grpc stream flow control window",
		"packet drop mitigation strategy",
		"mtu discovery and payload fragmentation",
		"connection pool max idle timeout",
		"service mesh proxy routing table",
		"tls certificate handshake timeout",
		"dns resolver query retry interval",
		"thread pool worker starvation limit",
		"rate limiter burst token bucket capacity",
		"write ahead log sync flush interval",
		"cluster quorum election timeout",
	}
	variations := []string{"config", "spec", "rule"}

	var nodes216 []model.Node
	var gannetryIDs []string
	var distractorIDs []string

	// 54 Gannetry Target Nodes
	for _, top := range topics {
		for _, v := range variations {
			id := fmt.Sprintf("gannetry-%s-%s", strings.ReplaceAll(top, " ", "-"), v)
			label := fmt.Sprintf("Project Gannetry %s %s", v, top)
			summary := fmt.Sprintf("Project Gannetry production %s for %s", v, top)
			emb := embedding.Generate(label+" "+summary, 384)
			gannetryIDs = append(gannetryIDs, id)
			nodes216 = append(nodes216, model.Node{
				ID:             id,
				EntityType:     "decision",
				Label:          label,
				Summary:        summary,
				Embedding:      emb,
				Anchors:        []string{"#project:gannetry"},
				CreatedAt:      now.Add(-24 * time.Hour),
				LastAccessedAt: now.Add(-30 * time.Minute),
				AccessCount:    35,
				StabilityScore: 1.0,
			})
		}
	}

	// 162 Distractor / Decoy Nodes (Tern, Cormorant, Generic)
	for _, proj := range []string{"Tern", "Cormorant", "Generic"} {
		for _, top := range topics {
			for _, v := range variations {
				id := fmt.Sprintf("dist-%s-%s-%s", strings.ToLower(proj), strings.ReplaceAll(top, " ", "-"), v)
				label := fmt.Sprintf("%s System %s %s", proj, v, top)
				summary := fmt.Sprintf("%s daemon %s for %s with port timeout and subgoal_of scheduler", proj, v, top)
				emb := embedding.Generate(label+" "+summary, 384)
				distractorIDs = append(distractorIDs, id)
				nodes216 = append(nodes216, model.Node{
					ID:             id,
					EntityType:     "concept",
					Label:          label,
					Summary:        summary,
					Embedding:      emb,
					CreatedAt:      now.Add(-6 * time.Hour),
					LastAccessedAt: now.Add(-5 * time.Minute),
					AccessCount:    45,
					StabilityScore: 1.0,
				})
			}
		}
	}

	var edges216 []model.Edge
	edgeSet216 := make(map[string]bool)
	addEdge216 := func(src, tgt, rel string, weight float64) {
		if src == tgt {
			return
		}
		k := fmt.Sprintf("%s->%s:%s", src, tgt, rel)
		if !edgeSet216[k] {
			edgeSet216[k] = true
			edges216 = append(edges216, model.Edge{
				SourceID:     src,
				TargetID:     tgt,
				RelationType: rel,
				Weight:       weight,
				CreatedAt:    now,
			})
		}
	}

	for i := 0; i < len(gannetryIDs) && len(edges216) < 250; i++ {
		tgt := gannetryIDs[(i+1)%len(gannetryIDs)]
		addEdge216(gannetryIDs[i], tgt, "depends_on", 0.85)
		tgt2 := gannetryIDs[(i+2)%len(gannetryIDs)]
		addEdge216(gannetryIDs[i], tgt2, "subgoal_of", 0.80)
	}
	for len(edges216) < 250 {
		src := gannetryIDs[rng.Intn(len(gannetryIDs))]
		tgt := gannetryIDs[rng.Intn(len(gannetryIDs))]
		addEdge216(src, tgt, "anchored_to", 0.75)
	}
	for len(edges216) < 550 {
		src := gannetryIDs[rng.Intn(len(gannetryIDs))]
		tgt := distractorIDs[rng.Intn(len(distractorIDs))]
		addEdge216(src, tgt, "incidental_mention", 0.40)
	}
	for len(edges216) < 866 {
		src := distractorIDs[rng.Intn(len(distractorIDs))]
		tgt := distractorIDs[rng.Intn(len(distractorIDs))]
		addEdge216(src, tgt, "relates_to", 0.55)
	}

	if _, err := store216.InsertNodes(ctx, nodes216); err != nil {
		fmt.Printf("[FATAL] Error inserting 216 nodes: %v\n", err)
		os.Exit(1)
	}
	if _, err := store216.InsertEdges(ctx, edges216); err != nil {
		fmt.Printf("[FATAL] Error inserting 866 edges: %v\n", err)
		os.Exit(1)
	}

	engine216, err := recall.NewEngine(ctx, store216, recall.DefaultConfig())
	if err != nil {
		fmt.Printf("[FATAL] Failed to initialise 216 engine: %v\n", err)
		os.Exit(1)
	}

	queryIterations := 100
	type testQuery struct {
		q        string
		targetID string
	}
	var queries []testQuery
	for i := 0; i < queryIterations; i++ {
		top := topics[i%len(topics)]
		v := variations[(i/len(topics))%len(variations)]
		q := fmt.Sprintf("Project Gannetry %s %s", v, top)
		expectedID := fmt.Sprintf("gannetry-%s-%s", strings.ReplaceAll(top, " ", "-"), v)
		queries = append(queries, testQuery{q: q, targetID: expectedID})
	}

	// 1. Expansion Disabled
	expFalse := false
	disabledHits := 0
	for _, tq := range queries {
		resp, err := engine216.Recall(ctx, model.RecallRequest{
			Query:            tq.q,
			TopK:             5,
			ExpandNeighbours: &expFalse,
		})
		if err != nil {
			fmt.Printf("[FATAL] Disabled recall failed: %v\n", err)
			os.Exit(1)
		}
		if len(resp.Nodes) > 0 && resp.Nodes[0].ID == tq.targetID {
			disabledHits++
		}
	}
	disabledHitRate := float64(disabledHits) / float64(queryIterations) * 100.0

	// 2. Gated 1-Hop Expansion
	expTrue := true
	minWeight := 0.60
	attnFactor := 0.35
	gatedHits := 0
	latencies216 := make([]float64, 0, queryIterations)
	var totalLat216 float64

	for _, tq := range queries {
		resp, err := engine216.Recall(ctx, model.RecallRequest{
			Query:             tq.q,
			TopK:              5,
			ExpandNeighbours:  &expTrue,
			MinEdgeWeight:     &minWeight,
			TraverseRelations: []string{"depends_on", "subgoal_of", "anchored_to"},
			AttenuationFactor: &attnFactor,
		})
		if err != nil {
			fmt.Printf("[FATAL] Gated recall failed: %v\n", err)
			os.Exit(1)
		}
		latencies216 = append(latencies216, resp.QueryLatencyMS)
		totalLat216 += resp.QueryLatencyMS

		if len(resp.Nodes) > 0 && resp.Nodes[0].ID == tq.targetID {
			gatedHits++
			if resp.Nodes[0].HopDistance != 0 {
				fmt.Printf("[FATAL] Rank 1 direct target must have hop_distance 0, got %d\n", resp.Nodes[0].HopDistance)
				os.Exit(1)
			}
		}
	}

	sort.Float64s(latencies216)
	meanLat216 := totalLat216 / float64(len(latencies216))
	p95Lat216 := latencies216[len(latencies216)*95/100]
	gatedHitRate := float64(gatedHits) / float64(queryIterations) * 100.0

	// 3. Ungated 1-Hop Expansion
	zeroWeight := 0.0
	highAttn := 0.70
	ungatedHits := 0
	for _, tq := range queries {
		resp, err := engine216.Recall(ctx, model.RecallRequest{
			Query:             tq.q,
			TopK:              5,
			ExpandNeighbours:  &expTrue,
			MinEdgeWeight:     &zeroWeight,
			AttenuationFactor: &highAttn,
		})
		if err != nil {
			fmt.Printf("[FATAL] Ungated recall failed: %v\n", err)
			os.Exit(1)
		}
		if len(resp.Nodes) > 0 && resp.Nodes[0].ID == tq.targetID {
			ungatedHits++
		}
	}
	ungatedHitRate := float64(ungatedHits) / float64(queryIterations) * 100.0

	fmt.Println("----------------------------------------------------------------")
	fmt.Println("     216-NODE / 866-EDGE GRAPH EXPANSION BENCHMARK RESULTS      ")
	fmt.Println("----------------------------------------------------------------")
	fmt.Printf(" Nodes / Edges:                  %d nodes / %d edges\n", len(nodes216), len(edges216))
	fmt.Printf(" Expansion Disabled Hit Rate:    %.1f%%\n", disabledHitRate)
	fmt.Printf(" Ungated Expansion Hit Rate:     %.1f%% (distractor dilution demonstrated)\n", ungatedHitRate)
	fmt.Printf(" Gated 1-Hop Expansion Hit Rate: %.1f%% (Target: >= 90.0%%)\n", gatedHitRate)
	fmt.Printf(" Mean Query Latency:             %.3f ms (Target: < 5.0 ms)\n", meanLat216)
	fmt.Printf(" 95th Percentile (p95):          %.3f ms (Target: < 5.0 ms)\n", p95Lat216)
	fmt.Println("----------------------------------------------------------------")

	if gatedHitRate < 90.0 || meanLat216 >= 5.0 {
		fmt.Printf("[FATAL] 216-node benchmark failed criteria: hit_rate=%.1f%%, mean_lat=%.3fms\n", gatedHitRate, meanLat216)
		os.Exit(1)
	}

	// Part B: Scaled 1,000-Node / 4,000-Edge Graph Benchmark
	fmt.Println("\n[Part B] Scaling Benchmark to 1,000 Nodes / 4,000 Relational Edges...")
	dbPath1000 := "test_benchmark_gated_1000.db"
	_ = os.Remove(dbPath1000)
	_ = os.Remove(dbPath1000 + "-wal")
	_ = os.Remove(dbPath1000 + "-shm")
	defer func() {
		_ = os.Remove(dbPath1000)
		_ = os.Remove(dbPath1000 + "-wal")
		_ = os.Remove(dbPath1000 + "-shm")
	}()

	store1000, err := store.NewSQLiteStore(dbPath1000)
	if err != nil {
		fmt.Printf("[FATAL] Failed to initialise 1000-node store: %v\n", err)
		os.Exit(1)
	}
	defer store1000.Close()

	rng1000 := rand.New(rand.NewSource(1337))
	topics1000 := append(topics, "raft log compaction threshold", "wal snapshot interval seconds")
	variations1000 := []string{"config", "spec", "rule", "deployment", "runtime"}

	var nodes1000 []model.Node
	var targetIDs1000 []string
	var distractorIDs1000 []string

	for _, top := range topics1000 {
		for _, v := range variations1000 {
			id := fmt.Sprintf("gannetry-1000-%s-%s", strings.ReplaceAll(top, " ", "-"), v)
			label := fmt.Sprintf("Project Gannetry %s %s", v, top)
			summary := fmt.Sprintf("Production configuration for Project Gannetry covering %s (%s)", top, v)
			emb := embedding.Generate(label+" "+summary, 384)
			targetIDs1000 = append(targetIDs1000, id)
			nodes1000 = append(nodes1000, model.Node{
				ID:             id,
				EntityType:     "decision",
				Label:          label,
				Summary:        summary,
				Embedding:      emb,
				Anchors:        []string{"#project:gannetry"},
				CreatedAt:      now.Add(-24 * time.Hour),
				LastAccessedAt: now.Add(-30 * time.Minute),
				AccessCount:    35,
				StabilityScore: 1.0,
			})
		}
	}

	decoys := []string{"Tern", "Cormorant", "Shearwater", "Petrel", "Guillemot", "Puffin", "Fulmar", "Auk", "Kittiwake"}
	for _, proj := range decoys {
		for _, top := range topics1000 {
			for _, v := range variations1000 {
				id := fmt.Sprintf("dist-1000-%s-%s-%s", strings.ToLower(proj), strings.ReplaceAll(top, " ", "-"), v)
				label := fmt.Sprintf("%s System %s %s", proj, v, top)
				summary := fmt.Sprintf("%s cluster daemon %s for %s with telemetry subgoal_of scheduler", proj, v, top)
				emb := embedding.Generate(label+" "+summary, 384)
				distractorIDs1000 = append(distractorIDs1000, id)
				nodes1000 = append(nodes1000, model.Node{
					ID:             id,
					EntityType:     "concept",
					Label:          label,
					Summary:        summary,
					Embedding:      emb,
					CreatedAt:      now.Add(-6 * time.Hour),
					LastAccessedAt: now.Add(-5 * time.Minute),
					AccessCount:    45,
					StabilityScore: 1.0,
				})
			}
		}
	}

	var edges1000 []model.Edge
	edgeSet1000 := make(map[string]bool)
	addEdge1000 := func(src, tgt, rel string, weight float64) {
		if src == tgt {
			return
		}
		k := fmt.Sprintf("%s->%s:%s", src, tgt, rel)
		if !edgeSet1000[k] {
			edgeSet1000[k] = true
			edges1000 = append(edges1000, model.Edge{
				SourceID:     src,
				TargetID:     tgt,
				RelationType: rel,
				Weight:       weight,
				CreatedAt:    now,
			})
		}
	}

	for i := 0; i < len(targetIDs1000) && len(edges1000) < 1000; i++ {
		tgt := targetIDs1000[(i+1)%len(targetIDs1000)]
		addEdge1000(targetIDs1000[i], tgt, "depends_on", 0.85)
		tgt2 := targetIDs1000[(i+3)%len(targetIDs1000)]
		addEdge1000(targetIDs1000[i], tgt2, "subgoal_of", 0.80)
	}
	for len(edges1000) < 1000 {
		src := targetIDs1000[rng1000.Intn(len(targetIDs1000))]
		tgt := targetIDs1000[rng1000.Intn(len(targetIDs1000))]
		addEdge1000(src, tgt, "anchored_to", 0.75)
	}
	for len(edges1000) < 2500 {
		src := targetIDs1000[rng1000.Intn(len(targetIDs1000))]
		tgt := distractorIDs1000[rng1000.Intn(len(distractorIDs1000))]
		addEdge1000(src, tgt, "incidental_mention", 0.40)
	}
	for len(edges1000) < 4000 {
		src := distractorIDs1000[rng1000.Intn(len(distractorIDs1000))]
		tgt := distractorIDs1000[rng1000.Intn(len(distractorIDs1000))]
		addEdge1000(src, tgt, "relates_to", 0.50+rng1000.Float64()*0.30)
	}

	if _, err := store1000.InsertNodes(ctx, nodes1000); err != nil {
		fmt.Printf("[FATAL] Error inserting 1000 nodes: %v\n", err)
		os.Exit(1)
	}
	if _, err := store1000.InsertEdges(ctx, edges1000); err != nil {
		fmt.Printf("[FATAL] Error inserting 4000 edges: %v\n", err)
		os.Exit(1)
	}

	engine1000, err := recall.NewEngine(ctx, store1000, recall.DefaultConfig())
	if err != nil {
		fmt.Printf("[FATAL] Failed to initialise 1000 engine: %v\n", err)
		os.Exit(1)
	}

	var queries1000 []testQuery
	for i := 0; i < queryIterations; i++ {
		top := topics1000[i%len(topics1000)]
		v := variations1000[(i/len(topics1000))%len(variations1000)]
		q := fmt.Sprintf("Project Gannetry %s %s", v, top)
		expectedID := fmt.Sprintf("gannetry-1000-%s-%s", strings.ReplaceAll(top, " ", "-"), v)
		queries1000 = append(queries1000, testQuery{q: q, targetID: expectedID})
	}

	gatedHits1000 := 0
	latencies1000 := make([]float64, 0, queryIterations)
	var totalLat1000 float64

	for _, tq := range queries1000 {
		resp, err := engine1000.Recall(ctx, model.RecallRequest{
			Query:             tq.q,
			TopK:              5,
			ExpandNeighbours:  &expTrue,
			MinEdgeWeight:     &minWeight,
			TraverseRelations: []string{"depends_on", "subgoal_of", "anchored_to"},
			AttenuationFactor: &attnFactor,
		})
		if err != nil {
			fmt.Printf("[FATAL] Gated 1000 recall query failed: %v\n", err)
			os.Exit(1)
		}
		latencies1000 = append(latencies1000, resp.QueryLatencyMS)
		totalLat1000 += resp.QueryLatencyMS

		if len(resp.Nodes) > 0 && resp.Nodes[0].ID == tq.targetID {
			gatedHits1000++
		}
	}

	sort.Float64s(latencies1000)
	meanLat1000 := totalLat1000 / float64(len(latencies1000))
	p95Lat1000 := latencies1000[len(latencies1000)*95/100]
	hitRate1000 := float64(gatedHits1000) / float64(queryIterations) * 100.0

	fmt.Println("----------------------------------------------------------------")
	fmt.Println("   1,000-NODE / 4,000-EDGE SCALED GRAPH BENCHMARK RESULTS       ")
	fmt.Println("----------------------------------------------------------------")
	fmt.Printf(" Nodes / Edges:                  %d nodes / %d edges\n", len(nodes1000), len(edges1000))
	fmt.Printf(" Gated 1-Hop Expansion Hit Rate: %.1f%% (Target: >= 90.0%%)\n", hitRate1000)
	fmt.Printf(" Mean Query Latency:             %.3f ms (Target: < 5.0 ms)\n", meanLat1000)
	fmt.Printf(" 95th Percentile (p95):          %.3f ms (Target: < 5.0 ms)\n", p95Lat1000)
	fmt.Println("----------------------------------------------------------------")

	if hitRate1000 < 90.0 || meanLat1000 >= 5.0 {
		fmt.Printf("[FATAL] 1,000-node benchmark failed criteria: hit_rate=%.1f%%, mean_lat=%.3fms\n", hitRate1000, meanLat1000)
		os.Exit(1)
	}

	// Part C: Unit Criteria Verification (Gating Threshold, Relation Filtering, Safe Defaults)
	fmt.Println("\n[Part C] Verifying Individual Traversal Gating Rules...")

	// Criterion 1: Threshold Gating (w=0.45 excluded when min=0.60)
	tWeight := 0.60
	respThresh, err := engine216.Recall(ctx, model.RecallRequest{
		Query:            "Project Gannetry config network port timeout settings",
		TopK:             10,
		ExpandNeighbours: &expTrue,
		MinEdgeWeight:    &tWeight,
	})
	if err != nil {
		fmt.Printf("[FATAL] Threshold gating check failed: %v\n", err)
		os.Exit(1)
	}
	for _, n := range respThresh.Nodes {
		if strings.HasPrefix(n.ID, "dist-") && n.HopDistance == 1 {
			fmt.Printf("[FATAL] Incidental edge (w=0.40) traversed when min_edge_weight=0.60!\n")
			os.Exit(1)
		}
	}
	fmt.Println(" [Rule 1] Edge Weight Gating: Incidental edges (w < 0.60) strictly excluded - PASS")

	// Criterion 2: Relation Type Whitelisting (traverse_relations = ["anchored_to"])
	respRel, err := engine216.Recall(ctx, model.RecallRequest{
		Query:             "Project Gannetry config network port timeout settings",
		TopK:              10,
		ExpandNeighbours:  &expTrue,
		MinEdgeWeight:     &minWeight,
		TraverseRelations: []string{"anchored_to"},
	})
	if err != nil {
		fmt.Printf("[FATAL] Relation whitelist check failed: %v\n", err)
		os.Exit(1)
	}
	for _, e := range respRel.Edges {
		if e.RelationType != "anchored_to" {
			fmt.Printf("[FATAL] Non-whitelisted edge relation %q traversed - expected only anchored_to\n", e.RelationType)
			os.Exit(1)
		}
	}
	fmt.Println(" [Rule 2] Relation Type Whitelisting: Only whitelisted relations traversed - PASS")

	// Criterion 3: Default Safe Behaviour (expand_neighbours omitted -> no expansion)
	respDef, err := engine216.Recall(ctx, model.RecallRequest{
		Query: "Project Gannetry config network port timeout settings",
		TopK:  10,
	})
	if err != nil {
		fmt.Printf("[FATAL] Default safe behaviour check failed: %v\n", err)
		os.Exit(1)
	}
	if len(respDef.Edges) > 0 {
		fmt.Printf("[FATAL] Default recall loaded %d edges when expand_neighbours was omitted!\n", len(respDef.Edges))
		os.Exit(1)
	}
	for _, n := range respDef.Nodes {
		if n.HopDistance != 0 {
			fmt.Printf("[FATAL] Default recall contained 1-hop neighbours when expand_neighbours was omitted!\n")
			os.Exit(1)
		}
	}
	fmt.Println(" [Rule 3] Default Safe Behaviour: 0 graph neighbours traversed when omitted - PASS")
}
