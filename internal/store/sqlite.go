package store

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	_ "modernc.org/sqlite"
)

// SQLiteStore implements the Store interface using pure-Go SQLite (modernc.org/sqlite).
type SQLiteStore struct {
	db     *sql.DB
	dbPath string
}

// NewSQLiteStore initialises and verifies the SQLite database schema at the specified path.
func NewSQLiteStore(dbPath string) (*SQLiteStore, error) {
	if dbPath == "" {
		dbPath = "sekha_knowledge.db"
	}

	dsn := fmt.Sprintf("%s?_busy_timeout=5000", dbPath)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite db: %w", err)
	}

	// Configure connection pool for concurrent reads
	db.SetMaxOpenConns(16)
	db.SetMaxIdleConns(8)
	db.SetConnMaxLifetime(time.Hour)

	store := &SQLiteStore{
		db:     db,
		dbPath: dbPath,
	}

	if err := store.initPragmasAndSchema(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("failed to initialise schema: %w", err)
	}

	return store, nil
}

func (s *SQLiteStore) initPragmasAndSchema() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	pragmas := []string{
		"PRAGMA journal_mode=WAL;",
		"PRAGMA synchronous=NORMAL;",
		"PRAGMA foreign_keys=ON;",
		"PRAGMA temp_store=MEMORY;",
		"PRAGMA cache_size=-64000;", // 64MB page cache
	}
	for _, pragma := range pragmas {
		if _, err := s.db.ExecContext(ctx, pragma); err != nil {
			return fmt.Errorf("error executing %s: %w", pragma, err)
		}
	}

	schema := `
	CREATE TABLE IF NOT EXISTS nodes (
		id TEXT PRIMARY KEY,
		entity_type TEXT NOT NULL,
		label TEXT NOT NULL,
		summary TEXT NOT NULL,
		embedding BLOB,
		created_at TIMESTAMP NOT NULL,
		last_accessed_at TIMESTAMP NOT NULL,
		access_count INTEGER NOT NULL DEFAULT 0,
		stability_score REAL NOT NULL DEFAULT 1.0
	);

	CREATE INDEX IF NOT EXISTS idx_nodes_entity_type ON nodes(entity_type);
	CREATE INDEX IF NOT EXISTS idx_nodes_last_accessed ON nodes(last_accessed_at);
	CREATE INDEX IF NOT EXISTS idx_nodes_access_count ON nodes(access_count);

	CREATE TABLE IF NOT EXISTS edges (
		source_id TEXT NOT NULL,
		target_id TEXT NOT NULL,
		relation_type TEXT NOT NULL,
		weight REAL NOT NULL DEFAULT 1.0,
		created_at TIMESTAMP NOT NULL,
		PRIMARY KEY (source_id, target_id, relation_type)
	);

	CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
	CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
	CREATE INDEX IF NOT EXISTS idx_edges_weight ON edges(weight);
	`

	if _, err := s.db.ExecContext(ctx, schema); err != nil {
		return fmt.Errorf("failed to create tables: %w", err)
	}

	return nil
}

func (s *SQLiteStore) Close() error {
	return s.db.Close()
}

// InsertNodes upserts nodes into the SQLite nodes table.
func (s *SQLiteStore) InsertNodes(ctx context.Context, nodes []model.Node) (int, error) {
	if len(nodes) == 0 {
		return 0, nil
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, err
	}
	defer func() { _ = tx.Rollback() }()

	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO nodes (id, entity_type, label, summary, embedding, created_at, last_accessed_at, access_count, stability_score)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			entity_type = excluded.entity_type,
			label = excluded.label,
			summary = excluded.summary,
			embedding = coalesce(excluded.embedding, nodes.embedding),
			stability_score = excluded.stability_score
	`)
	if err != nil {
		return 0, err
	}
	defer stmt.Close()

	now := time.Now().UTC()
	inserted := 0

	for _, n := range nodes {
		if n.ID == "" {
			continue
		}
		createdAt := n.CreatedAt
		if createdAt.IsZero() {
			createdAt = now
		}
		lastAccessed := n.LastAccessedAt
		if lastAccessed.IsZero() {
			lastAccessed = now
		}
		stability := n.StabilityScore
		if stability <= 0 {
			stability = 1.0
		}

		blob := EncodeEmbedding(n.Embedding)

		_, err := stmt.ExecContext(ctx,
			n.ID,
			n.EntityType,
			n.Label,
			n.Summary,
			blob,
			createdAt.Format(time.RFC3339Nano),
			lastAccessed.Format(time.RFC3339Nano),
			n.AccessCount,
			stability,
		)
		if err != nil {
			return inserted, fmt.Errorf("failed inserting node %s: %w", n.ID, err)
		}
		inserted++
	}

	if err := tx.Commit(); err != nil {
		return 0, err
	}
	return inserted, nil
}

// InsertEdges upserts edges into the SQLite edges table.
func (s *SQLiteStore) InsertEdges(ctx context.Context, edges []model.Edge) (int, error) {
	if len(edges) == 0 {
		return 0, nil
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, err
	}
	defer func() { _ = tx.Rollback() }()

	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO edges (source_id, target_id, relation_type, weight, created_at)
		VALUES (?, ?, ?, ?, ?)
		ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
			weight = excluded.weight,
			created_at = excluded.created_at
	`)
	if err != nil {
		return 0, err
	}
	defer stmt.Close()

	now := time.Now().UTC()
	inserted := 0

	for _, e := range edges {
		if e.SourceID == "" || e.TargetID == "" {
			continue
		}
		relType := e.RelationType
		if relType == "" {
			relType = "relates_to"
		}
		weight := e.Weight
		if weight <= 0 {
			weight = 1.0
		}
		createdAt := e.CreatedAt
		if createdAt.IsZero() {
			createdAt = now
		}

		_, err := stmt.ExecContext(ctx,
			e.SourceID,
			e.TargetID,
			relType,
			weight,
			createdAt.Format(time.RFC3339Nano),
		)
		if err != nil {
			return inserted, fmt.Errorf("failed inserting edge %s->%s: %w", e.SourceID, e.TargetID, err)
		}
		inserted++
	}

	if err := tx.Commit(); err != nil {
		return 0, err
	}
	return inserted, nil
}

// GetNode retrieves a single node by its ID.
func (s *SQLiteStore) GetNode(ctx context.Context, id string) (*model.Node, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT id, entity_type, label, summary, embedding, created_at, last_accessed_at, access_count, stability_score
		FROM nodes WHERE id = ?
	`, id)

	var n model.Node
	var blob []byte
	var createdStr, accessedStr string

	err := row.Scan(
		&n.ID,
		&n.EntityType,
		&n.Label,
		&n.Summary,
		&blob,
		&createdStr,
		&accessedStr,
		&n.AccessCount,
		&n.StabilityScore,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, err
	}

	n.Embedding = DecodeEmbedding(blob)
	if t, err := time.Parse(time.RFC3339Nano, createdStr); err == nil {
		n.CreatedAt = t
	}
	if t, err := time.Parse(time.RFC3339Nano, accessedStr); err == nil {
		n.LastAccessedAt = t
	}

	return &n, nil
}

// GetNodes retrieves multiple nodes by IDs.
func (s *SQLiteStore) GetNodes(ctx context.Context, ids []string) (map[string]model.Node, error) {
	res := make(map[string]model.Node)
	if len(ids) == 0 {
		return res, nil
	}

	// Process in batches of 500 to stay well under SQLite parameter limits
	const batchSize = 500
	for i := 0; i < len(ids); i += batchSize {
		end := i + batchSize
		if end > len(ids) {
			end = len(ids)
		}
		batch := ids[i:end]

		placeholders := strings.Repeat("?,", len(batch))
		placeholders = placeholders[:len(placeholders)-1]

		args := make([]any, len(batch))
		for idx, id := range batch {
			args[idx] = id
		}

		query := fmt.Sprintf(`
			SELECT id, entity_type, label, summary, embedding, created_at, last_accessed_at, access_count, stability_score
			FROM nodes WHERE id IN (%s)
		`, placeholders)

		rows, err := s.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, err
		}

		for rows.Next() {
			var n model.Node
			var blob []byte
			var createdStr, accessedStr string

			if err := rows.Scan(
				&n.ID,
				&n.EntityType,
				&n.Label,
				&n.Summary,
				&blob,
				&createdStr,
				&accessedStr,
				&n.AccessCount,
				&n.StabilityScore,
			); err != nil {
				rows.Close()
				return nil, err
			}

			n.Embedding = DecodeEmbedding(blob)
			if t, err := time.Parse(time.RFC3339Nano, createdStr); err == nil {
				n.CreatedAt = t
			}
			if t, err := time.Parse(time.RFC3339Nano, accessedStr); err == nil {
				n.LastAccessedAt = t
			}
			res[n.ID] = n
		}
		rows.Close()
	}

	return res, nil
}

// GetAllNodeHeaders scans lightweight headers and embeddings for in-memory indexing.
func (s *SQLiteStore) GetAllNodeHeaders(ctx context.Context) ([]NodeHeader, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, entity_type, embedding, last_accessed_at, access_count, stability_score
		FROM nodes
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var headers []NodeHeader
	for rows.Next() {
		var h NodeHeader
		var blob []byte
		var accessedStr string

		if err := rows.Scan(
			&h.ID,
			&h.EntityType,
			&blob,
			&accessedStr,
			&h.AccessCount,
			&h.StabilityScore,
		); err != nil {
			return nil, err
		}

		h.Embedding = DecodeEmbedding(blob)
		if len(h.Embedding) > 0 {
			var sum float64
			for _, v := range h.Embedding {
				sum += float64(v * v)
			}
			h.Magnitude = float32(math.Sqrt(sum))
		}
		if t, err := time.Parse(time.RFC3339Nano, accessedStr); err == nil {
			h.LastAccessedAt = t
		}
		headers = append(headers, h)
	}

	return headers, nil
}

// GetEdgesForNodes finds all edges originating from or pointing to any of the specified node IDs.
func (s *SQLiteStore) GetEdgesForNodes(ctx context.Context, nodeIDs []string) ([]model.Edge, error) {
	if len(nodeIDs) == 0 {
		return nil, nil
	}

	const batchSize = 500
	var allEdges []model.Edge

	for i := 0; i < len(nodeIDs); i += batchSize {
		end := i + batchSize
		if end > len(nodeIDs) {
			end = len(nodeIDs)
		}
		batch := nodeIDs[i:end]

		placeholders := strings.Repeat("?,", len(batch))
		placeholders = placeholders[:len(placeholders)-1]

		args := make([]any, len(batch)*2)
		for idx, id := range batch {
			args[idx] = id
			args[len(batch)+idx] = id
		}

		query := fmt.Sprintf(`
			SELECT source_id, target_id, relation_type, weight, created_at
			FROM edges
			WHERE source_id IN (%s) OR target_id IN (%s)
		`, placeholders, placeholders)

		rows, err := s.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, err
		}

		for rows.Next() {
			var e model.Edge
			var createdStr string
			if err := rows.Scan(&e.SourceID, &e.TargetID, &e.RelationType, &e.Weight, &createdStr); err != nil {
				rows.Close()
				return nil, err
			}
			if t, err := time.Parse(time.RFC3339Nano, createdStr); err == nil {
				e.CreatedAt = t
			}
			allEdges = append(allEdges, e)
		}
		rows.Close()
	}

	return allEdges, nil
}

// RecordAccess atomically increments access count and updates last_accessed_at.
func (s *SQLiteStore) RecordAccess(ctx context.Context, nodeIDs []string, accessTime time.Time) error {
	if len(nodeIDs) == 0 {
		return nil
	}

	timeStr := accessTime.UTC().Format(time.RFC3339Nano)
	const batchSize = 500

	for i := 0; i < len(nodeIDs); i += batchSize {
		end := i + batchSize
		if end > len(nodeIDs) {
			end = len(nodeIDs)
		}
		batch := nodeIDs[i:end]

		placeholders := strings.Repeat("?,", len(batch))
		placeholders = placeholders[:len(placeholders)-1]

		args := make([]any, 0, len(batch)+1)
		args = append(args, timeStr)
		for _, id := range batch {
			args = append(args, id)
		}

		query := fmt.Sprintf(`
			UPDATE nodes
			SET access_count = access_count + 1,
			    last_accessed_at = ?
			WHERE id IN (%s)
		`, placeholders)

		if _, err := s.db.ExecContext(ctx, query, args...); err != nil {
			return err
		}
	}

	return nil
}

// GetCounts returns total node and edge counts.
func (s *SQLiteStore) GetCounts(ctx context.Context) (int64, int64, error) {
	var nodeCount, edgeCount int64
	if err := s.db.QueryRowContext(ctx, "SELECT count(*) FROM nodes").Scan(&nodeCount); err != nil {
		return 0, 0, err
	}
	if err := s.db.QueryRowContext(ctx, "SELECT count(*) FROM edges").Scan(&edgeCount); err != nil {
		return nodeCount, 0, err
	}
	return nodeCount, edgeCount, nil
}

// GetGraphSummary calculates topology metrics including degree and density.
func (s *SQLiteStore) GetGraphSummary(ctx context.Context) (*model.GraphSummary, error) {
	nodeCount, edgeCount, err := s.GetCounts(ctx)
	if err != nil {
		return nil, err
	}

	entityTypes := make(map[string]int64)
	rows, err := s.db.QueryContext(ctx, "SELECT entity_type, count(*) FROM nodes GROUP BY entity_type")
	if err == nil {
		for rows.Next() {
			var et string
			var count int64
			if err := rows.Scan(&et, &count); err == nil {
				entityTypes[et] = count
			}
		}
		rows.Close()
	}

	var avgDegree float64
	var density float64
	if nodeCount > 0 {
		avgDegree = float64(edgeCount*2) / float64(nodeCount)
		if nodeCount > 1 {
			// Directed graph maximum edges = N * (N - 1)
			maxPossibleEdges := float64(nodeCount * (nodeCount - 1))
			density = float64(edgeCount) / maxPossibleEdges
		}
	}

	var dbSize int64
	if s.dbPath != "" && s.dbPath != ":memory:" {
		if fi, err := os.Stat(s.dbPath); err == nil {
			dbSize = fi.Size()
		}
	}

	return &model.GraphSummary{
		NodeCount:    nodeCount,
		EdgeCount:    edgeCount,
		EntityTypes:  entityTypes,
		AvgDegree:    avgDegree,
		GraphDensity: density,
		DBSizeBytes:  dbSize,
	}, nil
}
