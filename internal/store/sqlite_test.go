package store

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
)

func newTestSQLiteStore(t *testing.T) *SQLiteStore {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test_anchors.db")

	s, err := NewSQLiteStore(dbPath)
	if err != nil {
		t.Fatalf("failed to create sqlite store: %v", err)
	}

	t.Cleanup(func() {
		_ = s.Close()
		_ = os.Remove(dbPath)
	})

	return s
}

func TestSQLiteStore_AnchorPersistenceAndRetrieval(t *testing.T) {
	ctx := context.Background()
	s := newTestSQLiteStore(t)

	now := time.Now().UTC()
	node := model.Node{
		ID:             "node-kestrel-1",
		EntityType:     "decision",
		Label:          "Kestrel Ingest Daemon Port",
		Summary:        "Configured port 8084 for Project Kestrel",
		Embedding:      []float32{0.1, 0.2, 0.3},
		CreatedAt:      now,
		LastAccessedAt: now,
		StabilityScore: 1.0,
	}

	// 1. Insert node without anchors
	inserted, err := s.InsertNodes(ctx, []model.Node{node})
	if err != nil || inserted != 1 {
		t.Fatalf("failed inserting node: %v", err)
	}

	// 2. Attach anchors with variations and duplicates
	anchorsToAttach := []string{
		"#project:kestrel",
		"pattern:circuit-breaker", // should normalize to #pattern:circuit-breaker
		" #PROJECT:KESTREL ",      // duplicate, case/whitespace variant
		"#auth:jwt",
	}
	if err := s.AttachAnchors(ctx, node.ID, anchorsToAttach); err != nil {
		t.Fatalf("failed attaching anchors: %v", err)
	}

	// 3. Retrieve anchors for node
	retrievedAnchors, err := s.GetAnchorsForNode(ctx, node.ID)
	if err != nil {
		t.Fatalf("failed getting anchors: %v", err)
	}

	expectedCount := 3 // #auth:jwt, #pattern:circuit-breaker, #project:kestrel
	if len(retrievedAnchors) != expectedCount {
		t.Fatalf("expected %d anchors, got %d: %v", expectedCount, len(retrievedAnchors), retrievedAnchors)
	}

	anchorMap := make(map[string]bool)
	for _, a := range retrievedAnchors {
		anchorMap[a] = true
	}
	for _, exp := range []string{"#auth:jwt", "#pattern:circuit-breaker", "#project:kestrel"} {
		if !anchorMap[exp] {
			t.Errorf("missing expected anchor tag: %s", exp)
		}
	}

	// 4. Test GetNodeIDsForAnchors
	nodeIDs, err := s.GetNodeIDsForAnchors(ctx, []string{"#project:kestrel"})
	if err != nil {
		t.Fatalf("failed getting node IDs for anchor: %v", err)
	}
	if len(nodeIDs) != 1 || nodeIDs[0] != node.ID {
		t.Fatalf("expected node IDs [%s], got %v", node.ID, nodeIDs)
	}

	// 5. Test GetNode populates anchors
	fetchedNode, err := s.GetNode(ctx, node.ID)
	if err != nil {
		t.Fatalf("GetNode failed: %v", err)
	}
	if fetchedNode == nil {
		t.Fatalf("expected fetched node, got nil")
	}
	if len(fetchedNode.Anchors) != expectedCount {
		t.Fatalf("expected %d anchors on fetched node, got %d", expectedCount, len(fetchedNode.Anchors))
	}

	// 6. Test GetNodes populates anchors
	nodesMap, err := s.GetNodes(ctx, []string{node.ID})
	if err != nil {
		t.Fatalf("GetNodes failed: %v", err)
	}
	if len(nodesMap[node.ID].Anchors) != expectedCount {
		t.Fatalf("expected %d anchors in GetNodes, got %d", expectedCount, len(nodesMap[node.ID].Anchors))
	}

	// 7. Test GetAllNodeHeaders populates anchors
	headers, err := s.GetAllNodeHeaders(ctx)
	if err != nil {
		t.Fatalf("GetAllNodeHeaders failed: %v", err)
	}
	if len(headers) != 1 {
		t.Fatalf("expected 1 header, got %d", len(headers))
	}
	if len(headers[0].Anchors) != expectedCount {
		t.Fatalf("expected %d anchors on header, got %d", expectedCount, len(headers[0].Anchors))
	}
}

func TestSQLiteStore_InsertNodesWithAnchors(t *testing.T) {
	ctx := context.Background()
	s := newTestSQLiteStore(t)

	now := time.Now().UTC()
	nodes := []model.Node{
		{
			ID:             "node-falcon-1",
			EntityType:     "concept",
			Label:          "Falcon Timeout Config",
			Summary:        "Timeout configs for Project Falcon",
			Anchors:        []string{"#project:falcon", "#env:prod"},
			CreatedAt:      now,
			LastAccessedAt: now,
			StabilityScore: 1.0,
		},
		{
			ID:             "node-kestrel-2",
			EntityType:     "concept",
			Label:          "Kestrel Timeout Config",
			Summary:        "Timeout configs for Project Kestrel",
			Anchors:        []string{"#project:kestrel", "#env:prod"},
			CreatedAt:      now,
			LastAccessedAt: now,
			StabilityScore: 1.0,
		},
	}

	inserted, err := s.InsertNodes(ctx, nodes)
	if err != nil || inserted != 2 {
		t.Fatalf("InsertNodes failed: %v", err)
	}

	// Query IDs for #project:falcon
	falconIDs, err := s.GetNodeIDsForAnchors(ctx, []string{"#project:falcon"})
	if err != nil {
		t.Fatalf("GetNodeIDsForAnchors failed: %v", err)
	}
	if len(falconIDs) != 1 || falconIDs[0] != "node-falcon-1" {
		t.Fatalf("expected [node-falcon-1], got %v", falconIDs)
	}

	// Query IDs for shared #env:prod
	prodIDs, err := s.GetNodeIDsForAnchors(ctx, []string{"#env:prod"})
	if err != nil {
		t.Fatalf("GetNodeIDsForAnchors failed: %v", err)
	}
	if len(prodIDs) != 2 {
		t.Fatalf("expected 2 nodes with #env:prod, got %d: %v", len(prodIDs), prodIDs)
	}
}

func TestSQLiteStore_CascadeDeletion(t *testing.T) {
	ctx := context.Background()
	s := newTestSQLiteStore(t)

	now := time.Now().UTC()
	node := model.Node{
		ID:             "node-to-delete",
		EntityType:     "concept",
		Label:          "Ephemeral Node",
		Summary:        "Will be deleted",
		Anchors:        []string{"#project:ephemeral"},
		CreatedAt:      now,
		LastAccessedAt: now,
		StabilityScore: 1.0,
	}

	_, err := s.InsertNodes(ctx, []model.Node{node})
	if err != nil {
		t.Fatalf("InsertNodes failed: %v", err)
	}

	anchors, err := s.GetAnchorsForNode(ctx, node.ID)
	if err != nil || len(anchors) != 1 {
		t.Fatalf("expected 1 anchor before deletion, got %v", anchors)
	}

	// Delete node directly from nodes table
	_, err = s.db.ExecContext(ctx, "DELETE FROM nodes WHERE id = ?", node.ID)
	if err != nil {
		t.Fatalf("failed deleting node: %v", err)
	}

	// Check that node_anchors entries were cascade deleted
	anchorsAfter, err := s.GetAnchorsForNode(ctx, node.ID)
	if err != nil {
		t.Fatalf("GetAnchorsForNode failed after deletion: %v", err)
	}
	if len(anchorsAfter) != 0 {
		t.Fatalf("expected 0 anchors after cascade delete, got %v", anchorsAfter)
	}
}

func TestSQLiteStore_SchemaIdempotence(t *testing.T) {
	s := newTestSQLiteStore(t)

	// Running initPragmasAndSchema multiple times should not error
	for i := 0; i < 3; i++ {
		if err := s.initPragmasAndSchema(); err != nil {
			t.Fatalf("iteration %d: initPragmasAndSchema failed: %v", i, err)
		}
	}
}
