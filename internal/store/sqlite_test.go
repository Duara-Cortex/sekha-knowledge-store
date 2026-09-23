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

func TestSQLiteStore_ImportanceScorePersistence(t *testing.T) {
	ctx := context.Background()
	s := newTestSQLiteStore(t)

	now := time.Now().UTC()
	node := model.Node{
		ID:              "node-imp-1",
		EntityType:      "concept",
		Label:           "Cluster Master Config",
		Summary:         "Ingest port and master bind rules",
		ImportanceScore: 0.92,
		StabilityScore:  1.0,
		CreatedAt:       now,
		LastAccessedAt:  now,
	}

	inserted, err := s.InsertNodes(ctx, []model.Node{node})
	if err != nil || inserted != 1 {
		t.Fatalf("InsertNodes failed: %v", err)
	}

	// 1. GetNode
	fetched, err := s.GetNode(ctx, node.ID)
	if err != nil || fetched == nil {
		t.Fatalf("GetNode failed: %v", err)
	}
	if fetched.ImportanceScore != 0.92 {
		t.Errorf("expected ImportanceScore 0.92, got %f", fetched.ImportanceScore)
	}

	// 2. GetNodes
	m, err := s.GetNodes(ctx, []string{node.ID})
	if err != nil || m[node.ID].ImportanceScore != 0.92 {
		t.Errorf("GetNodes failed or incorrect importance: %v, got %f", err, m[node.ID].ImportanceScore)
	}

	// 3. GetAllNodeHeaders
	headers, err := s.GetAllNodeHeaders(ctx)
	if err != nil || len(headers) != 1 || headers[0].ImportanceScore != 0.92 {
		t.Errorf("GetAllNodeHeaders failed or incorrect importance: %v, got %f", err, headers[0].ImportanceScore)
	}

	// 4. GetAllActiveNodes
	actives, err := s.GetAllActiveNodes(ctx)
	if err != nil || len(actives) != 1 || actives[0].ImportanceScore != 0.92 {
		t.Errorf("GetAllActiveNodes failed or incorrect importance: %v, got %f", err, actives[0].ImportanceScore)
	}

	// 5. UpsertNode with higher importance
	node.ImportanceScore = 0.98
	if err := s.UpsertNode(ctx, node); err != nil {
		t.Fatalf("UpsertNode failed: %v", err)
	}

	fetched2, err := s.GetNode(ctx, node.ID)
	if err != nil || fetched2.ImportanceScore != 0.98 {
		t.Errorf("UpsertNode did not update importance: got %f", fetched2.ImportanceScore)
	}
}

func TestSQLiteStore_SchemaMigrationWithExistingNodes(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "legacy_migration.db")

	// 1. Create table without importance_score column (legacy schema)
	rawStore, err := NewSQLiteStore(dbPath)
	if err != nil {
		t.Fatalf("initial store create failed: %v", err)
	}
	_ = rawStore.Close()

	// 2. Open raw db, drop importance_score if possible or create a legacy table
	// SQLite ALTER TABLE cannot drop columns in very old versions, so we recreate the legacy table
	legacyDB, err := openRawDB(dbPath)
	if err != nil {
		t.Fatalf("openRawDB failed: %v", err)
	}
	_, err = legacyDB.Exec(`
		DROP TABLE IF EXISTS nodes;
		CREATE TABLE nodes (
			id TEXT PRIMARY KEY,
			entity_type TEXT NOT NULL,
			label TEXT NOT NULL,
			summary TEXT NOT NULL,
			embedding BLOB,
			created_at TEXT NOT NULL,
			last_accessed_at TEXT NOT NULL,
			last_reinforced_at TEXT,
			access_count INTEGER NOT NULL DEFAULT 1,
			stability_score REAL NOT NULL DEFAULT 1.0,
			is_archived INTEGER NOT NULL DEFAULT 0,
			archived_at TEXT
		);
		INSERT INTO nodes (id, entity_type, label, summary, created_at, last_accessed_at, access_count, stability_score)
		VALUES ('legacy-node-1', 'concept', 'Legacy Node', 'Pre-existing node', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 1, 1.0);
	`)
	if err != nil {
		legacyDB.Close()
		t.Fatalf("failed creating legacy schema: %v", err)
	}
	legacyDB.Close()

	// 3. Open with NewSQLiteStore which should execute ALTER TABLE nodes ADD COLUMN importance_score
	migratedStore, err := NewSQLiteStore(dbPath)
	if err != nil {
		t.Fatalf("failed reopening store with migration: %v", err)
	}
	defer migratedStore.Close()

	// Verify legacy node read has default 0.5 importance_score
	legacyNode, err := migratedStore.GetNode(ctx, "legacy-node-1")
	if err != nil {
		t.Fatalf("failed reading legacy node: %v", err)
	}
	if legacyNode.ImportanceScore != 0.5 {
		t.Errorf("expected default importance 0.5 for migrated node, got %f", legacyNode.ImportanceScore)
	}

	// Verify inserting new node with importance_score succeeds
	newNode := model.Node{
		ID:              "new-node-1",
		EntityType:      "decision",
		Label:           "Architecture Decision",
		Summary:         "New architecture decision",
		ImportanceScore: 0.88,
		StabilityScore:  1.0,
		CreatedAt:       time.Now().UTC(),
		LastAccessedAt:  time.Now().UTC(),
	}
	if _, err := migratedStore.InsertNodes(ctx, []model.Node{newNode}); err != nil {
		t.Fatalf("failed inserting into migrated table: %v", err)
	}
	fetchedNew, err := migratedStore.GetNode(ctx, "new-node-1")
	if err != nil || fetchedNew.ImportanceScore != 0.88 {
		t.Errorf("failed fetching newly inserted node from migrated store: %v", err)
	}
}

func openRawDB(path string) (*sqlDBWrapper, error) {
	// Simple helper to open raw database for schema testing
	s, err := NewSQLiteStore(path)
	if err != nil {
		return nil, err
	}
	return &sqlDBWrapper{s}, nil
}

type sqlDBWrapper struct {
	*SQLiteStore
}

func (w *sqlDBWrapper) Exec(query string, args ...any) (any, error) {
	return w.db.Exec(query, args...)
}

func (w *sqlDBWrapper) Close() error {
	return w.SQLiteStore.Close()
}

func TestSQLiteStore_ApplyScopedDecay(t *testing.T) {
	ctx := context.Background()
	s := newTestSQLiteStore(t)
	now := time.Now().UTC()

	nodes := []model.Node{
		{
			ID:              "node-distractor-1",
			EntityType:      "telemetry",
			Label:           "transient sensor jitter code_123",
			Summary:         "Session sess-alpha debug event",
			ImportanceScore: 0.25,
			StabilityScore:  0.30,
			CreatedAt:       now.Add(-2 * time.Hour),
			LastAccessedAt:  now.Add(-2 * time.Hour),
		},
		{
			ID:              "node-distractor-2",
			EntityType:      "noise",
			Label:           "ephemeral memory spike",
			Summary:         "Session sess-beta telemetry noise",
			ImportanceScore: 0.20,
			StabilityScore:  0.30,
			CreatedAt:       now.Add(-2 * time.Hour),
			LastAccessedAt:  now.Add(-2 * time.Hour),
		},
		{
			ID:              "node-anchored-hub",
			EntityType:      "concept",
			Label:           "Cluster Coordinator Hub",
			Summary:         "Session sess-alpha hub node",
			Anchors:         []string{"#project:kestrel"},
			ImportanceScore: 0.40, // Low importance, but protected by anchor!
			StabilityScore:  0.30,
			CreatedAt:       now.Add(-2 * time.Hour),
			LastAccessedAt:  now.Add(-2 * time.Hour),
		},
		{
			ID:              "node-important-rule",
			EntityType:      "decision",
			Label:           "Ingest Port Binding Rule",
			Summary:         "Session sess-alpha important rule",
			ImportanceScore: 0.90, // Unanchored, but protected by importance >= 0.80!
			StabilityScore:  0.30,
			CreatedAt:       now.Add(-2 * time.Hour),
			LastAccessedAt:  now.Add(-2 * time.Hour),
		},
	}

	_, err := s.InsertNodes(ctx, nodes)
	if err != nil {
		t.Fatalf("InsertNodes failed: %v", err)
	}

	// Insert edge between distractor-1 and anchored-hub
	edges := []model.Edge{
		{
			SourceID:     "node-distractor-1",
			TargetID:     "node-anchored-hub",
			RelationType: "relates_to",
			Weight:       1.0,
			CreatedAt:    now.Add(-2 * time.Hour),
		},
	}
	_, err = s.InsertEdges(ctx, edges)
	if err != nil {
		t.Fatalf("InsertEdges failed: %v", err)
	}

	// 1. Test Session Scoping ("session" with SessionID: "sess-beta")
	zeroGrace := 0 * time.Second
	oneHour := 1 * time.Hour
	pruneThresh := 0.50

	_, archived, _, protected, err := s.ApplyScopedDecay(ctx, model.DecayRequest{
		Scope:                 "session",
		SessionID:             "sess-beta",
		InactivityGracePeriod: &zeroGrace,
		DecayHalfLife:         &oneHour,
		PruneThreshold:        &pruneThresh,
		MinImportanceToRetain: 0.80,
	}, now)
	if err != nil {
		t.Fatalf("ApplyScopedDecay failed for session scope: %v", err)
	}

	// Protected count: node-anchored-hub (has anchor) + node-important-rule (imp=0.90) = 2
	if protected < 2 {
		t.Errorf("expected at least 2 protected nodes, got %d", protected)
	}
	// Archived: only node-distractor-2 belongs to sess-beta
	if archived != 1 {
		t.Errorf("expected 1 node archived in sess-beta, got %d", archived)
	}

	// Verify distractor-2 is archived, distractor-1 is still active
	d2, _ := s.GetNode(ctx, "node-distractor-2")
	if d2 == nil || !d2.IsArchived {
		t.Errorf("expected node-distractor-2 to be archived")
	}
	d1, _ := s.GetNode(ctx, "node-distractor-1")
	if d1 == nil || d1.IsArchived {
		t.Errorf("expected node-distractor-1 to remain active after sess-beta decay")
	}

	// 2. Test Unanchored Scoping on remaining active nodes with 0s grace period
	_, archived, prunedEdges, protected, err := s.ApplyScopedDecay(ctx, model.DecayRequest{
		Scope:                 "unanchored",
		InactivityGracePeriod: &zeroGrace,
		DecayHalfLife:         &oneHour,
		PruneThreshold:        &pruneThresh,
		MinImportanceToRetain: 0.80,
	}, now)
	if err != nil {
		t.Fatalf("ApplyScopedDecay failed for unanchored scope: %v", err)
	}

	// distractor-1 should now be archived
	if archived != 1 {
		t.Errorf("expected 1 node archived in unanchored decay, got %d", archived)
	}
	d1After, _ := s.GetNode(ctx, "node-distractor-1")
	if d1After == nil || !d1After.IsArchived {
		t.Errorf("expected node-distractor-1 to be archived")
	}

	// node-anchored-hub MUST remain active (protected by anchor)
	hub, _ := s.GetNode(ctx, "node-anchored-hub")
	if hub == nil || hub.IsArchived {
		t.Errorf("node-anchored-hub was improperly archived")
	}

	// node-important-rule MUST remain active (protected by importance >= 0.80)
	rule, _ := s.GetNode(ctx, "node-important-rule")
	if rule == nil || rule.IsArchived {
		t.Errorf("node-important-rule was improperly archived")
	}

	// Incident edge from distractor-1 should have been pruned
	if prunedEdges < 1 {
		t.Errorf("expected edge connected to distractor-1 to be pruned, got %d", prunedEdges)
	}
}
