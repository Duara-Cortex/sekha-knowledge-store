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

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/recall"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

func main() {
	nodeCount := flag.Int("nodes", 10000, "Number of synthetic knowledge nodes to generate")
	edgeCount := flag.Int("edges", 25000, "Number of synthetic relational edges to generate")
	dimensions := flag.Int("dims", 64, "Embedding vector dimensionality")
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
		// Normalise vector to unit length
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

		elapsedHours := rng.Float64() * 720.0 // up to 30 days old
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
	fmt.Println("[Step 3] Initialising in-memory associative recall engine...")
	hydrateStart := time.Now()
	engine, err := recall.NewEngine(ctx, sqliteStore, recall.DefaultConfig())
	if err != nil {
		fmt.Printf("[FATAL] Failed to initialise recall engine: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("         Hydrated index for %d nodes in %.2fms\n", *nodeCount, float64(time.Since(hydrateStart).Microseconds())/1000.0)

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
		// Generate random query vector
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

	targetLatency := 20.0 // Target is < 20ms per Task 08 requirements

	fmt.Println("----------------------------------------------------------------")
	fmt.Println("                    EMPIRICAL BENCHMARK RESULTS                 ")
	fmt.Println("----------------------------------------------------------------")
	fmt.Printf(" Total Synthetic Nodes: %d\n", *nodeCount)
	fmt.Printf(" Total Relational Edges: %d\n", *edgeCount)
	fmt.Printf(" Mean Recall Latency:   %.3f ms (Target: < %.1f ms)\n", mean, targetLatency)
	fmt.Printf(" Median (p50) Latency:  %.3f ms\n", p50)
	fmt.Printf(" 90th Percentile (p90): %.3f ms\n", p90)
	fmt.Printf(" 95th Percentile (p95): %.3f ms (Target: < %.1f ms)\n", p95, targetLatency)
	fmt.Printf(" 99th Percentile (p99): %.3f ms\n", p99)
	fmt.Printf(" Min / Max Latency:     %.3f ms / %.3f ms\n", minLat, maxLat)
	fmt.Println("----------------------------------------------------------------")

	if p95 < targetLatency && mean < targetLatency {
		fmt.Printf(" RESULT: PASS - Associative recall latency on %d nodes\n", *nodeCount)
		fmt.Printf(" satisfies the sub-20ms requirement by a factor of %.1fx!\n", targetLatency/mean)
	} else {
		fmt.Printf(" RESULT: FAIL - Recall latency exceeded %.1fms threshold\n", targetLatency)
		os.Exit(1)
	}
	fmt.Println("================================================================")
}
