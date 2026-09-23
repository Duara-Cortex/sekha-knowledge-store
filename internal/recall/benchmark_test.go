package recall

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/embedding"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

// TestMultiProjectSyntheticBenchmark evaluates first-query hit rate and query latency
// across multiple projects sharing overlapping generic tokens (e.g. "port", "timeout", "buffer", "retry").
func TestMultiProjectSyntheticBenchmark(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "benchmark_multi_project.db")

	sqliteStore, err := store.NewSQLiteStore(dbPath)
	if err != nil {
		t.Fatalf("failed to create sqlite store: %v", err)
	}
	t.Cleanup(func() {
		_ = sqliteStore.Close()
		_ = os.Remove(dbPath)
	})

	rng := rand.New(rand.NewSource(1337))

	projects := []struct {
		name   string
		anchor string
	}{
		{name: "Kestrel", anchor: "#project:kestrel"},
		{name: "Falcon", anchor: "#project:falcon"},
		{name: "Merlin", anchor: "#project:merlin"},
		{name: "Osprey", anchor: "#project:osprey"},
	}

	genericTopics := []string{
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
	}

	now := time.Now().UTC()
	var allNodes []model.Node

	// Generate nodes for each project using identical generic topics
	nodeIDSeq := 0
	for _, proj := range projects {
		for topicIdx, topic := range genericTopics {
			nodeIDSeq++
			// Vary recency and access counts across projects so non-target projects can have higher recency/frequency
			recencyHours := rng.Float64() * 120.0
			accessCount := int64(rng.Intn(100))

			// Falcon nodes have unnaturally high access counts and recency to act as aggressive distractors
			if proj.name == "Falcon" {
				recencyHours = rng.Float64() * 1.0 // very recent
				accessCount = int64(80 + rng.Intn(50))
			}

			lastAccessed := now.Add(-time.Duration(recencyHours * float64(time.Hour)))

			allNodes = append(allNodes, model.Node{
				ID:             fmt.Sprintf("node-proj-%d-%d", nodeIDSeq, topicIdx),
				EntityType:     "concept",
				Label:          fmt.Sprintf("%s %s", proj.name, topic),
				Summary:        fmt.Sprintf("Detailed configuration parameters for %s covering %s with specific bounds.", proj.name, topic),
				Anchors:        []string{proj.anchor, "#layer:infrastructure"},
				CreatedAt:      lastAccessed.Add(-24 * time.Hour),
				LastAccessedAt: lastAccessed,
				AccessCount:    accessCount,
				StabilityScore: 0.8 + rng.Float64()*0.2,
			})
		}
	}

	// Add generic distractor nodes without project anchors
	for i := 0; i < 50; i++ {
		nodeIDSeq++
		topic := genericTopics[i%len(genericTopics)]
		allNodes = append(allNodes, model.Node{
			ID:             fmt.Sprintf("node-distractor-%d", nodeIDSeq),
			EntityType:     "telemetry",
			Label:          fmt.Sprintf("Generic System %s", topic),
			Summary:        fmt.Sprintf("Telemetry logs tracking %s across all cluster daemons.", topic),
			Anchors:        []string{"#env:staging"},
			CreatedAt:      now.Add(-time.Duration(rng.Intn(500)) * time.Minute),
			LastAccessedAt: now.Add(-time.Duration(rng.Intn(50)) * time.Minute),
			AccessCount:    int64(rng.Intn(120)),
			StabilityScore: 1.0,
		})
	}

	// Ingest all nodes into SQLite store
	if _, err := sqliteStore.InsertNodes(ctx, allNodes); err != nil {
		t.Fatalf("failed inserting synthetic nodes: %v", err)
	}

	// Initialize recall engine
	engine, err := NewEngine(ctx, sqliteStore, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	// Run benchmark queries
	queryIterations := 200
	hitCount := 0
	latencies := make([]float64, 0, queryIterations)
	var totalLatency float64

	for i := 0; i < queryIterations; i++ {
		targetProj := projects[rng.Intn(len(projects))]
		topic := genericTopics[rng.Intn(len(genericTopics))]

		// The query string contains only the generic terms, with the project specified via target anchor
		req := model.RecallRequest{
			Query:      topic,
			Anchors:    []string{targetProj.anchor},
			AnchorMode: "boost",
			TopK:       5,
		}

		resp, err := engine.Recall(ctx, req)
		if err != nil {
			t.Fatalf("query %d failed: %v", i, err)
		}

		if len(resp.Nodes) == 0 {
			t.Fatalf("query %d returned 0 nodes", i)
		}

		latencies = append(latencies, resp.QueryLatencyMS)
		totalLatency += resp.QueryLatencyMS

		// Check if Rank 1 node has the target anchor
		rank1Node := resp.Nodes[0]
		hasTargetAnchor := false
		for _, a := range rank1Node.Anchors {
			if a == targetProj.anchor {
				hasTargetAnchor = true
				break
			}
		}

		if hasTargetAnchor {
			hitCount++
		}
	}

	sort.Float64s(latencies)
	meanLat := totalLatency / float64(len(latencies))
	p50 := latencies[len(latencies)*50/100]
	p90 := latencies[len(latencies)*90/100]
	p95 := latencies[len(latencies)*95/100]
	p99 := latencies[len(latencies)*99/100]
	hitRate := float64(hitCount) / float64(queryIterations) * 100.0

	t.Logf("================================================================")
	t.Logf("   MULTI-PROJECT SYNTHETIC BENCHMARK EMPIRICAL RESULTS          ")
	t.Logf("================================================================")
	t.Logf(" Total Synthetic Nodes:   %d (across %d projects)", len(allNodes), len(projects))
	t.Logf(" Query Iterations:        %d", queryIterations)
	t.Logf(" Target Rank 1 Hit Rate:  %.1f%% (Required: >= 90.0%%)", hitRate)
	t.Logf(" Mean Latency:            %.3f ms (Required: < 20.0 ms)", meanLat)
	t.Logf(" Median (p50) Latency:    %.3f ms", p50)
	t.Logf(" 90th Percentile (p90):   %.3f ms", p90)
	t.Logf(" 95th Percentile (p95):   %.3f ms (Required: < 20.0 ms)", p95)
	t.Logf(" 99th Percentile (p99):   %.3f ms", p99)
	t.Logf("================================================================")

	// Verification 1: Rank 1 first-query hit >= 90%
	if hitRate < 90.0 {
		t.Fatalf("Benchmark failed: Rank 1 hit rate was %.1f%%, expected >= 90.0%%", hitRate)
	}

	// Verification 2: Query latency remains within acceptable edge bounds (< 20ms)
	if p95 >= 20.0 || meanLat >= 20.0 {
		t.Fatalf("Benchmark failed: latency exceeded 20ms threshold (p95=%.3fms, mean=%.3fms)", p95, meanLat)
	}
}

// TestGated1HopGraphExpansion_216NodeBenchmark evaluates gated 1-hop graph expansion against a synthetic
// 216-node / 866-edge graph, measuring and comparing ungated, gated, and disabled recall modes.
func TestGated1HopGraphExpansion_216NodeBenchmark(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "benchmark_216.db")

	sqliteStore, err := store.NewSQLiteStore(dbPath)
	if err != nil {
		t.Fatalf("failed to create sqlite store: %v", err)
	}
	t.Cleanup(func() {
		_ = sqliteStore.Close()
		_ = os.Remove(dbPath)
	})

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

	var nodes []model.Node
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
			nodes = append(nodes, model.Node{
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

	// 162 Distractor / Decoy Nodes (Tern, Cormorant, Generic) sharing generic tokens
	for _, proj := range []string{"Tern", "Cormorant", "Generic"} {
		for _, top := range topics {
			for _, v := range variations {
				id := fmt.Sprintf("dist-%s-%s-%s", strings.ToLower(proj), strings.ReplaceAll(top, " ", "-"), v)
				label := fmt.Sprintf("%s System %s %s", proj, v, top)
				summary := fmt.Sprintf("%s daemon %s for %s with port timeout and subgoal_of scheduler", proj, v, top)
				emb := embedding.Generate(label+" "+summary, 384)
				distractorIDs = append(distractorIDs, id)
				nodes = append(nodes, model.Node{
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

	var edges []model.Edge
	edgeSet := make(map[string]bool)
	addEdge := func(src, tgt, rel string, weight float64) {
		if src == tgt {
			return
		}
		k := fmt.Sprintf("%s->%s:%s", src, tgt, rel)
		if !edgeSet[k] {
			edgeSet[k] = true
			edges = append(edges, model.Edge{
				SourceID:     src,
				TargetID:     tgt,
				RelationType: rel,
				Weight:       weight,
				CreatedAt:    now,
			})
		}
	}

	// 250 high-weight intra-Gannetry edges (w >= 0.70)
	for i := 0; i < len(gannetryIDs) && len(edges) < 250; i++ {
		tgt := gannetryIDs[(i+1)%len(gannetryIDs)]
		addEdge(gannetryIDs[i], tgt, "depends_on", 0.85)
		tgt2 := gannetryIDs[(i+2)%len(gannetryIDs)]
		addEdge(gannetryIDs[i], tgt2, "subgoal_of", 0.80)
	}
	for len(edges) < 250 {
		src := gannetryIDs[rng.Intn(len(gannetryIDs))]
		tgt := gannetryIDs[rng.Intn(len(gannetryIDs))]
		addEdge(src, tgt, "anchored_to", 0.75)
	}

	// 300 weak/incidental edges connecting Gannetry to distractors (w = 0.40)
	for len(edges) < 550 {
		src := gannetryIDs[rng.Intn(len(gannetryIDs))]
		tgt := distractorIDs[rng.Intn(len(distractorIDs))]
		addEdge(src, tgt, "incidental_mention", 0.40)
	}

	// 316 inter-distractor edges
	for len(edges) < 866 {
		src := distractorIDs[rng.Intn(len(distractorIDs))]
		tgt := distractorIDs[rng.Intn(len(distractorIDs))]
		addEdge(src, tgt, "relates_to", 0.55)
	}

	if len(nodes) != 216 {
		t.Fatalf("expected exactly 216 nodes, got %d", len(nodes))
	}
	if len(edges) != 866 {
		t.Fatalf("expected exactly 866 edges, got %d", len(edges))
	}

	if _, err := sqliteStore.InsertNodes(ctx, nodes); err != nil {
		t.Fatalf("failed inserting nodes: %v", err)
	}
	if _, err := sqliteStore.InsertEdges(ctx, edges); err != nil {
		t.Fatalf("failed inserting edges: %v", err)
	}

	engine, err := NewEngine(ctx, sqliteStore, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	queryIterations := 100
	var queries []struct {
		q        string
		targetID string
	}
	for i := 0; i < queryIterations; i++ {
		top := topics[i%len(topics)]
		v := variations[(i/len(topics))%len(variations)]
		q := fmt.Sprintf("Project Gannetry %s %s", v, top)
		expectedID := fmt.Sprintf("gannetry-%s-%s", strings.ReplaceAll(top, " ", "-"), v)
		queries = append(queries, struct {
			q        string
			targetID string
		}{q: q, targetID: expectedID})
	}

	// 1. Expansion Disabled (expand_neighbours: false)
	expFalse := false
	disabledHits := 0
	for _, tq := range queries {
		resp, err := engine.Recall(ctx, model.RecallRequest{
			Query:            tq.q,
			TopK:             5,
			ExpandNeighbours: &expFalse,
		})
		if err != nil {
			t.Fatalf("disabled recall failed: %v", err)
		}
		if len(resp.Nodes) > 0 && resp.Nodes[0].ID == tq.targetID {
			disabledHits++
		}
	}
	disabledHitRate := float64(disabledHits) / float64(queryIterations) * 100.0

	// 2. Gated 1-Hop Expansion (min_edge_weight: 0.60, relations: ["depends_on", "subgoal_of", "anchored_to"], attenuation: 0.35)
	expTrue := true
	minWeight := 0.60
	attnFactor := 0.35
	gatedHits := 0
	latencies := make([]float64, 0, queryIterations)
	var totalLatency float64

	for _, tq := range queries {
		resp, err := engine.Recall(ctx, model.RecallRequest{
			Query:             tq.q,
			TopK:              5,
			ExpandNeighbours:  &expTrue,
			MinEdgeWeight:     &minWeight,
			TraverseRelations: []string{"depends_on", "subgoal_of", "anchored_to"},
			AttenuationFactor: &attnFactor,
		})
		if err != nil {
			t.Fatalf("gated recall failed: %v", err)
		}
		latencies = append(latencies, resp.QueryLatencyMS)
		totalLatency += resp.QueryLatencyMS

		if len(resp.Nodes) > 0 && resp.Nodes[0].ID == tq.targetID {
			gatedHits++
			if resp.Nodes[0].HopDistance != 0 {
				t.Fatalf("Rank 1 direct target must have hop_distance 0, got %d", resp.Nodes[0].HopDistance)
			}
		}
	}

	sort.Float64s(latencies)
	meanLat := totalLatency / float64(len(latencies))
	p50 := latencies[len(latencies)*50/100]
	p90 := latencies[len(latencies)*90/100]
	p95 := latencies[len(latencies)*95/100]
	p99 := latencies[len(latencies)*99/100]
	gatedHitRate := float64(gatedHits) / float64(queryIterations) * 100.0

	// 3. Ungated 1-Hop Expansion (min_edge_weight: 0.0, no relation filtering)
	zeroWeight := 0.0
	highAttn := 0.70
	ungatedHits := 0
	for _, tq := range queries {
		resp, err := engine.Recall(ctx, model.RecallRequest{
			Query:             tq.q,
			TopK:              5,
			ExpandNeighbours:  &expTrue,
			MinEdgeWeight:     &zeroWeight,
			AttenuationFactor: &highAttn,
		})
		if err != nil {
			t.Fatalf("ungated recall failed: %v", err)
		}
		if len(resp.Nodes) > 0 && resp.Nodes[0].ID == tq.targetID {
			ungatedHits++
		}
	}
	ungatedHitRate := float64(ungatedHits) / float64(queryIterations) * 100.0

	t.Logf("================================================================")
	t.Logf("   216-NODE / 866-EDGE GRAPH EXPANSION BENCHMARK RESULTS        ")
	t.Logf("================================================================")
	t.Logf(" Total Synthetic Nodes:          %d (54 target, 162 distractor)", len(nodes))
	t.Logf(" Total Relational Edges:         %d", len(edges))
	t.Logf(" Benchmark Queries:              %d iterations", queryIterations)
	t.Logf(" Expansion Disabled Hit Rate:    %.1f%%", disabledHitRate)
	t.Logf(" Gated 1-Hop Expansion Hit Rate: %.1f%% (Required: >= 90.0%%)", gatedHitRate)
	t.Logf(" Ungated Expansion Hit Rate:     %.1f%%", ungatedHitRate)
	t.Logf(" Mean Query Latency:             %.3f ms (Required: < 5.0 ms)", meanLat)
	t.Logf(" Median (p50) Latency:           %.3f ms", p50)
	t.Logf(" 90th Percentile (p90):          %.3f ms", p90)
	t.Logf(" 95th Percentile (p95):          %.3f ms", p95)
	t.Logf(" 99th Percentile (p99):          %.3f ms", p99)
	t.Logf("================================================================")

	// Verification 1: Gated target recall reaches >= 90.0% Rank 1 accuracy
	if gatedHitRate < 90.0 {
		t.Fatalf("Benchmark failed: gated hit rate was %.1f%%, expected >= 90.0%%", gatedHitRate)
	}

	// Verification 2: Mean query latency < 5.0 ms (or < 25.0 ms under Go race detector instrumentation)
	maxLatency := 5.0
	if raceDetectorEnabled {
		maxLatency = 25.0
	}
	if meanLat >= maxLatency {
		t.Fatalf("Benchmark failed: mean latency was %.3f ms, expected < %.1f ms", meanLat, maxLatency)
	}
}

// TestGated1HopGraphExpansion_Scaled1000NodeBenchmark evaluates gated 1-hop graph expansion
// on a scaled 1,000-node / 4,000-edge graph to assert sub-5ms query latency and >= 90% hit rate.
func TestGated1HopGraphExpansion_Scaled1000NodeBenchmark(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "benchmark_1000.db")

	sqliteStore, err := store.NewSQLiteStore(dbPath)
	if err != nil {
		t.Fatalf("failed to create sqlite store: %v", err)
	}
	t.Cleanup(func() {
		_ = sqliteStore.Close()
		_ = os.Remove(dbPath)
	})

	rng := rand.New(rand.NewSource(1337))
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
		"raft log compaction threshold",
		"wal snapshot interval seconds",
	}
	variations := []string{"config", "spec", "rule", "deployment", "runtime"}

	var nodes []model.Node
	var targetIDs []string
	var distractorIDs []string

	// 100 Target Nodes (Project Gannetry: 20 topics * 5 variations)
	for _, top := range topics {
		for _, v := range variations {
			id := fmt.Sprintf("gannetry-1000-%s-%s", strings.ReplaceAll(top, " ", "-"), v)
			label := fmt.Sprintf("Project Gannetry %s %s", v, top)
			summary := fmt.Sprintf("Production configuration for Project Gannetry covering %s (%s)", top, v)
			emb := embedding.Generate(label+" "+summary, 384)
			targetIDs = append(targetIDs, id)
			nodes = append(nodes, model.Node{
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

	// 900 Distractor Nodes across 9 decoy projects (900 nodes)
	decoys := []string{"Tern", "Cormorant", "Shearwater", "Petrel", "Guillemot", "Puffin", "Fulmar", "Auk", "Kittiwake"}
	for _, proj := range decoys {
		for _, top := range topics {
			for _, v := range variations {
				id := fmt.Sprintf("dist-1000-%s-%s-%s", strings.ToLower(proj), strings.ReplaceAll(top, " ", "-"), v)
				label := fmt.Sprintf("%s System %s %s", proj, v, top)
				summary := fmt.Sprintf("%s cluster daemon %s for %s with telemetry subgoal_of scheduler", proj, v, top)
				emb := embedding.Generate(label+" "+summary, 384)
				distractorIDs = append(distractorIDs, id)
				nodes = append(nodes, model.Node{
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

	var edges []model.Edge
	edgeSet := make(map[string]bool)
	addEdge := func(src, tgt, rel string, weight float64) {
		if src == tgt {
			return
		}
		k := fmt.Sprintf("%s->%s:%s", src, tgt, rel)
		if !edgeSet[k] {
			edgeSet[k] = true
			edges = append(edges, model.Edge{
				SourceID:     src,
				TargetID:     tgt,
				RelationType: rel,
				Weight:       weight,
				CreatedAt:    now,
			})
		}
	}

	// 1,000 valid high-weight intra-target edges (w >= 0.70)
	for i := 0; i < len(targetIDs) && len(edges) < 1000; i++ {
		tgt := targetIDs[(i+1)%len(targetIDs)]
		addEdge(targetIDs[i], tgt, "depends_on", 0.85)
		tgt2 := targetIDs[(i+3)%len(targetIDs)]
		addEdge(targetIDs[i], tgt2, "subgoal_of", 0.80)
	}
	for len(edges) < 1000 {
		src := targetIDs[rng.Intn(len(targetIDs))]
		tgt := targetIDs[rng.Intn(len(targetIDs))]
		addEdge(src, tgt, "anchored_to", 0.75)
	}

	// 1,500 weak/incidental edges connecting target to distractors (w = 0.40)
	for len(edges) < 2500 {
		src := targetIDs[rng.Intn(len(targetIDs))]
		tgt := distractorIDs[rng.Intn(len(distractorIDs))]
		addEdge(src, tgt, "incidental_mention", 0.40)
	}

	// 1,500 inter-distractor edges (up to 4,000 total edges)
	for len(edges) < 4000 {
		src := distractorIDs[rng.Intn(len(distractorIDs))]
		tgt := distractorIDs[rng.Intn(len(distractorIDs))]
		addEdge(src, tgt, "relates_to", 0.50+rng.Float64()*0.30)
	}

	if len(nodes) != 1000 {
		t.Fatalf("expected exactly 1000 nodes, got %d", len(nodes))
	}
	if len(edges) != 4000 {
		t.Fatalf("expected exactly 4000 edges, got %d", len(edges))
	}

	if _, err := sqliteStore.InsertNodes(ctx, nodes); err != nil {
		t.Fatalf("failed inserting nodes: %v", err)
	}
	if _, err := sqliteStore.InsertEdges(ctx, edges); err != nil {
		t.Fatalf("failed inserting edges: %v", err)
	}

	engine, err := NewEngine(ctx, sqliteStore, DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	queryIterations := 100
	var queries []struct {
		q        string
		targetID string
	}
	for i := 0; i < queryIterations; i++ {
		top := topics[i%len(topics)]
		v := variations[(i/len(topics))%len(variations)]
		q := fmt.Sprintf("Project Gannetry %s %s", v, top)
		expectedID := fmt.Sprintf("gannetry-1000-%s-%s", strings.ReplaceAll(top, " ", "-"), v)
		queries = append(queries, struct {
			q        string
			targetID string
		}{q: q, targetID: expectedID})
	}

	expTrue := true
	minWeight := 0.60
	attnFactor := 0.35
	gatedHits := 0
	latencies := make([]float64, 0, queryIterations)
	var totalLatency float64

	for _, tq := range queries {
		resp, err := engine.Recall(ctx, model.RecallRequest{
			Query:             tq.q,
			TopK:              5,
			ExpandNeighbours:  &expTrue,
			MinEdgeWeight:     &minWeight,
			TraverseRelations: []string{"depends_on", "subgoal_of", "anchored_to"},
			AttenuationFactor: &attnFactor,
		})
		if err != nil {
			t.Fatalf("gated recall query failed: %v", err)
		}
		latencies = append(latencies, resp.QueryLatencyMS)
		totalLatency += resp.QueryLatencyMS

		if len(resp.Nodes) > 0 && resp.Nodes[0].ID == tq.targetID {
			gatedHits++
		}
	}

	sort.Float64s(latencies)
	meanLat := totalLatency / float64(len(latencies))
	p50 := latencies[len(latencies)*50/100]
	p95 := latencies[len(latencies)*95/100]
	gatedHitRate := float64(gatedHits) / float64(queryIterations) * 100.0

	t.Logf("================================================================")
	t.Logf("   1,000-NODE / 4,000-EDGE SCALED GRAPH BENCHMARK RESULTS       ")
	t.Logf("================================================================")
	t.Logf(" Total Synthetic Nodes:          %d (100 target, 900 distractor)", len(nodes))
	t.Logf(" Total Relational Edges:         %d", len(edges))
	t.Logf(" Benchmark Queries:              %d iterations", queryIterations)
	t.Logf(" Gated 1-Hop Expansion Hit Rate: %.1f%% (Required: >= 90.0%%)", gatedHitRate)
	t.Logf(" Mean Query Latency:             %.3f ms (Required: < 5.0 ms)", meanLat)
	t.Logf(" Median (p50) Latency:           %.3f ms", p50)
	t.Logf(" 95th Percentile (p95):          %.3f ms", p95)
	t.Logf("================================================================")

	if gatedHitRate < 90.0 {
		t.Fatalf("Scaled benchmark failed: hit rate was %.1f%%, expected >= 90.0%%", gatedHitRate)
	}
	maxLatency := 5.0
	if raceDetectorEnabled {
		maxLatency = 25.0
	}
	if meanLat >= maxLatency {
		t.Fatalf("Scaled benchmark failed: mean latency was %.3f ms, expected < %.1f ms", meanLat, maxLatency)
	}
}
