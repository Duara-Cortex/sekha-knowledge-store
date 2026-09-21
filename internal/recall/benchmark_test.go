package recall

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

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
