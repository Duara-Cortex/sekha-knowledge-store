package store

import (
	"context"
	"database/sql"
	"encoding/json"
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
		last_reinforced_at TIMESTAMP,
		archived_at TIMESTAMP,
		access_count INTEGER NOT NULL DEFAULT 0,
		stability_score REAL NOT NULL DEFAULT 1.0,
		is_archived INTEGER NOT NULL DEFAULT 0
	);

	CREATE INDEX IF NOT EXISTS idx_nodes_entity_type ON nodes(entity_type);
	CREATE INDEX IF NOT EXISTS idx_nodes_last_accessed ON nodes(last_accessed_at);
	CREATE INDEX IF NOT EXISTS idx_nodes_access_count ON nodes(access_count);
	CREATE INDEX IF NOT EXISTS idx_nodes_archived ON nodes(is_archived, stability_score);

	CREATE TABLE IF NOT EXISTS edges (
		source_id TEXT NOT NULL,
		target_id TEXT NOT NULL,
		relation_type TEXT NOT NULL,
		weight REAL NOT NULL DEFAULT 1.0,
		created_at TIMESTAMP NOT NULL,
		last_reinforced_at TIMESTAMP,
		PRIMARY KEY (source_id, target_id, relation_type)
	);

	CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
	CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
	CREATE INDEX IF NOT EXISTS idx_edges_weight ON edges(weight);

	CREATE TABLE IF NOT EXISTS episodic_traces (
		id TEXT PRIMARY KEY,
		session_id TEXT NOT NULL,
		task_goal TEXT NOT NULL,
		outcome TEXT NOT NULL,
		trace_payload JSON NOT NULL,
		consolidated INTEGER NOT NULL DEFAULT 0,
		created_at TIMESTAMP NOT NULL,
		consolidated_at TIMESTAMP
	);

	CREATE INDEX IF NOT EXISTS idx_traces_consolidated ON episodic_traces(consolidated, created_at);
	`

	if _, err := s.db.ExecContext(ctx, schema); err != nil {
		return fmt.Errorf("failed to create tables: %w", err)
	}

	// Idempotent column migrations for backward compatibility with pre-existing databases
	migrations := []string{
		"ALTER TABLE nodes ADD COLUMN is_archived INTEGER NOT NULL DEFAULT 0;",
		"ALTER TABLE nodes ADD COLUMN archived_at TIMESTAMP;",
		"ALTER TABLE nodes ADD COLUMN last_reinforced_at TIMESTAMP;",
		"ALTER TABLE edges ADD COLUMN last_reinforced_at TIMESTAMP;",
	}
	for _, migration := range migrations {
		_, _ = s.db.ExecContext(ctx, migration)
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
		INSERT INTO nodes (id, entity_type, label, summary, embedding, created_at, last_accessed_at, last_reinforced_at, archived_at, access_count, stability_score, is_archived)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			entity_type = excluded.entity_type,
			label = excluded.label,
			summary = excluded.summary,
			embedding = coalesce(excluded.embedding, nodes.embedding),
			stability_score = excluded.stability_score,
			last_accessed_at = excluded.last_accessed_at,
			last_reinforced_at = coalesce(excluded.last_reinforced_at, nodes.last_reinforced_at),
			archived_at = excluded.archived_at,
			is_archived = excluded.is_archived
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

		var reinforcedStr sql.NullString
		if !n.LastReinforcedAt.IsZero() {
			reinforcedStr = sql.NullString{String: n.LastReinforcedAt.Format(time.RFC3339Nano), Valid: true}
		}

		var archivedStr sql.NullString
		if n.ArchivedAt != nil && !n.ArchivedAt.IsZero() {
			archivedStr = sql.NullString{String: n.ArchivedAt.Format(time.RFC3339Nano), Valid: true}
		}

		isArchivedInt := 0
		if n.IsArchived {
			isArchivedInt = 1
		}

		_, err := stmt.ExecContext(ctx,
			n.ID,
			n.EntityType,
			n.Label,
			n.Summary,
			blob,
			createdAt.Format(time.RFC3339Nano),
			lastAccessed.Format(time.RFC3339Nano),
			reinforcedStr,
			archivedStr,
			n.AccessCount,
			stability,
			isArchivedInt,
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
		INSERT INTO edges (source_id, target_id, relation_type, weight, created_at, last_reinforced_at)
		VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
			weight = excluded.weight,
			created_at = excluded.created_at,
			last_reinforced_at = coalesce(excluded.last_reinforced_at, edges.last_reinforced_at)
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

		var reinforcedStr sql.NullString
		if !e.LastReinforcedAt.IsZero() {
			reinforcedStr = sql.NullString{String: e.LastReinforcedAt.Format(time.RFC3339Nano), Valid: true}
		}

		_, err := stmt.ExecContext(ctx,
			e.SourceID,
			e.TargetID,
			relType,
			weight,
			createdAt.Format(time.RFC3339Nano),
			reinforcedStr,
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

func scanNodeFromRow(row interface{ Scan(dest ...any) error }) (*model.Node, error) {
	var n model.Node
	var blob []byte
	var createdStr, accessedStr string
	var reinforcedStr, archivedStr sql.NullString
	var isArchivedInt int

	err := row.Scan(
		&n.ID,
		&n.EntityType,
		&n.Label,
		&n.Summary,
		&blob,
		&createdStr,
		&accessedStr,
		&reinforcedStr,
		&archivedStr,
		&n.AccessCount,
		&n.StabilityScore,
		&isArchivedInt,
	)
	if err != nil {
		return nil, err
	}

	n.Embedding = DecodeEmbedding(blob)
	if t, err := time.Parse(time.RFC3339Nano, createdStr); err == nil {
		n.CreatedAt = t
	}
	if t, err := time.Parse(time.RFC3339Nano, accessedStr); err == nil {
		n.LastAccessedAt = t
	}
	if reinforcedStr.Valid {
		if t, err := time.Parse(time.RFC3339Nano, reinforcedStr.String); err == nil {
			n.LastReinforcedAt = t
		}
	}
	if archivedStr.Valid {
		if t, err := time.Parse(time.RFC3339Nano, archivedStr.String); err == nil {
			n.ArchivedAt = &t
		}
	}
	n.IsArchived = isArchivedInt == 1

	return &n, nil
}

// GetNode retrieves a single node by its ID.
func (s *SQLiteStore) GetNode(ctx context.Context, id string) (*model.Node, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT id, entity_type, label, summary, embedding, created_at, last_accessed_at, last_reinforced_at, archived_at, access_count, stability_score, is_archived
		FROM nodes WHERE id = ?
	`, id)

	n, err := scanNodeFromRow(row)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, err
	}

	return n, nil
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
			SELECT id, entity_type, label, summary, embedding, created_at, last_accessed_at, last_reinforced_at, archived_at, access_count, stability_score, is_archived
			FROM nodes WHERE id IN (%s)
		`, placeholders)

		rows, err := s.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, err
		}

		for rows.Next() {
			n, err := scanNodeFromRow(rows)
			if err != nil {
				rows.Close()
				return nil, err
			}
			res[n.ID] = *n
		}
		rows.Close()
	}

	return res, nil
}

// GetAllNodeHeaders scans lightweight headers and embeddings for in-memory indexing of active nodes.
func (s *SQLiteStore) GetAllNodeHeaders(ctx context.Context) ([]NodeHeader, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, entity_type, label, summary, embedding, last_accessed_at, access_count, stability_score, is_archived
		FROM nodes
		WHERE is_archived = 0
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
		var isArchivedInt int

		if err := rows.Scan(
			&h.ID,
			&h.EntityType,
			&h.Label,
			&h.Summary,
			&blob,
			&accessedStr,
			&h.AccessCount,
			&h.StabilityScore,
			&isArchivedInt,
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
		h.IsArchived = isArchivedInt == 1
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

// FindMatchingNode searches for an active, non-archived node matching the label and optional entity type.
func (s *SQLiteStore) FindMatchingNode(ctx context.Context, label string, entityType string) (*model.Node, error) {
	trimmed := strings.TrimSpace(label)
	if trimmed == "" {
		return nil, nil
	}

	query := `
		SELECT id, entity_type, label, summary, embedding, created_at, last_accessed_at, last_reinforced_at, archived_at, access_count, stability_score, is_archived
		FROM nodes
		WHERE lower(label) = lower(?) AND is_archived = 0
	`
	args := []any{trimmed}
	if entityType != "" {
		query += " AND entity_type = ?"
		args = append(args, entityType)
	}
	query += " ORDER BY stability_score DESC LIMIT 1"

	row := s.db.QueryRowContext(ctx, query, args...)
	n, err := scanNodeFromRow(row)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, err
	}
	return n, nil
}

// QueueTrace persists an episodic deliberation trace dispatched from Node 2 into the episodic_traces queue.
func (s *SQLiteStore) QueueTrace(ctx context.Context, trace model.EpisodicTrace) error {
	if trace.ID == "" {
		trace.ID = fmt.Sprintf("trace-%d", time.Now().UTC().UnixNano())
	}
	if trace.CreatedAt.IsZero() {
		trace.CreatedAt = time.Now().UTC()
	}
	if trace.Outcome == "" {
		trace.Outcome = model.OutcomeNeutral
	}

	payloadBytes, err := json.Marshal(trace)
	if err != nil {
		return fmt.Errorf("failed to serialise episodic trace payload: %w", err)
	}

	_, err = s.db.ExecContext(ctx, `
		INSERT INTO episodic_traces (id, session_id, task_goal, outcome, trace_payload, consolidated, created_at)
		VALUES (?, ?, ?, ?, ?, 0, ?)
		ON CONFLICT(id) DO UPDATE SET
			task_goal = excluded.task_goal,
			outcome = excluded.outcome,
			trace_payload = excluded.trace_payload
	`, trace.ID, trace.SessionID, trace.TaskGoal, trace.Outcome, string(payloadBytes), trace.CreatedAt.Format(time.RFC3339Nano))
	if err != nil {
		return fmt.Errorf("failed queueing episodic trace %s: %w", trace.ID, err)
	}
	return nil
}

// GetPendingTraces retrieves unconsolidated episodic traces up to the specified limit.
func (s *SQLiteStore) GetPendingTraces(ctx context.Context, limit int) ([]model.EpisodicTrace, error) {
	if limit <= 0 {
		limit = 100
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, session_id, task_goal, outcome, trace_payload, consolidated, created_at, consolidated_at
		FROM episodic_traces
		WHERE consolidated = 0
		ORDER BY created_at ASC
		LIMIT ?
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("failed querying pending traces: %w", err)
	}
	defer rows.Close()

	var traces []model.EpisodicTrace
	for rows.Next() {
		var t model.EpisodicTrace
		var payload string
		var consolidatedInt int
		var createdStr string
		var consolidatedStr sql.NullString

		if err := rows.Scan(
			&t.ID,
			&t.SessionID,
			&t.TaskGoal,
			&t.Outcome,
			&payload,
			&consolidatedInt,
			&createdStr,
			&consolidatedStr,
		); err != nil {
			return nil, fmt.Errorf("failed scanning episodic trace: %w", err)
		}

		if err := json.Unmarshal([]byte(payload), &t); err != nil {
			t.Consolidated = consolidatedInt == 1
		}
		if parsed, err := time.Parse(time.RFC3339Nano, createdStr); err == nil {
			t.CreatedAt = parsed
		}
		if consolidatedStr.Valid {
			if parsed, err := time.Parse(time.RFC3339Nano, consolidatedStr.String); err == nil {
				t.ConsolidatedAt = &parsed
			}
		}
		traces = append(traces, t)
	}
	return traces, nil
}

// MarkTraceConsolidated marks an episodic trace as successfully consolidated.
func (s *SQLiteStore) MarkTraceConsolidated(ctx context.Context, traceID string, consolidatedAt time.Time) error {
	if consolidatedAt.IsZero() {
		consolidatedAt = time.Now().UTC()
	}
	_, err := s.db.ExecContext(ctx, `
		UPDATE episodic_traces
		SET consolidated = 1,
		    consolidated_at = ?
		WHERE id = ?
	`, consolidatedAt.Format(time.RFC3339Nano), traceID)
	return err
}

// ReinforceEdge implements Hebbian reinforcement on directed edges between co-activated entities.
func (s *SQLiteStore) ReinforceEdge(ctx context.Context, sourceID, targetID, relationType string, deltaW float64, maxWeight float64, reinforcedAt time.Time) (float64, error) {
	if sourceID == "" || targetID == "" {
		return 0, fmt.Errorf("source and target IDs cannot be empty")
	}
	if relationType == "" {
		relationType = "relates_to"
	}
	if maxWeight <= 0 {
		maxWeight = 5.0
	}
	if reinforcedAt.IsZero() {
		reinforcedAt = time.Now().UTC()
	}

	timeStr := reinforcedAt.Format(time.RFC3339Nano)
	initialWeight := math.Min(maxWeight, 1.0+deltaW)

	row := s.db.QueryRowContext(ctx, `
		INSERT INTO edges (source_id, target_id, relation_type, weight, created_at, last_reinforced_at)
		VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
			weight = min(?, edges.weight + ?),
			last_reinforced_at = excluded.last_reinforced_at
		RETURNING weight
	`, sourceID, targetID, relationType, initialWeight, timeStr, timeStr, maxWeight, deltaW)

	var newWeight float64
	if err := row.Scan(&newWeight); err != nil {
		return 0, fmt.Errorf("failed reinforcing edge %s->%s: %w", sourceID, targetID, err)
	}
	return newWeight, nil
}

// BoostNodeStability reinforces a node's stability score and recency upon co-activation.
func (s *SQLiteStore) BoostNodeStability(ctx context.Context, nodeID string, deltaStability float64, reinforcedAt time.Time) error {
	if nodeID == "" {
		return nil
	}
	if reinforcedAt.IsZero() {
		reinforcedAt = time.Now().UTC()
	}

	timeStr := reinforcedAt.Format(time.RFC3339Nano)
	_, err := s.db.ExecContext(ctx, `
		UPDATE nodes
		SET stability_score = min(1.0, stability_score + ?),
		    access_count = access_count + 1,
		    last_accessed_at = ?,
		    last_reinforced_at = ?,
		    is_archived = 0,
		    archived_at = NULL
		WHERE id = ?
	`, deltaStability, timeStr, timeStr, nodeID)
	return err
}

// ApplyDecayAndPrune executes the mathematical recency decay function and soft-archives / prunes decaying traces.
func (s *SQLiteStore) ApplyDecayAndPrune(ctx context.Context, cfg model.DecayConfig, refTime time.Time) (decayed int, archived int, prunedEdges int, err error) {
	if refTime.IsZero() {
		refTime = time.Now().UTC()
	}
	if cfg.DecayHalfLife <= 0 {
		cfg.DecayHalfLife = 72 * time.Hour
	}
	if cfg.PruneThreshold <= 0 {
		cfg.PruneThreshold = 0.10
	}
	if cfg.EdgePruneThreshold <= 0 {
		cfg.EdgePruneThreshold = 0.05
	}
	if cfg.InactivityGracePeriod <= 0 {
		cfg.InactivityGracePeriod = 168 * time.Hour
	}

	// Decay constant lambda = ln(2) / tau
	lambda := math.Ln2 / cfg.DecayHalfLife.Seconds()

	// 1. Fetch active nodes for mathematical recency decay evaluation
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, created_at, last_accessed_at, last_reinforced_at, stability_score
		FROM nodes
		WHERE is_archived = 0
	`)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed querying active nodes for decay: %w", err)
	}

	type nodeDecayItem struct {
		id           string
		newStability float64
		shouldPrune  bool
	}
	var decayQueue []nodeDecayItem

	for rows.Next() {
		var id string
		var createdStr, accessedStr string
		var reinforcedStr sql.NullString
		var stability float64

		if err := rows.Scan(&id, &createdStr, &accessedStr, &reinforcedStr, &stability); err != nil {
			rows.Close()
			return 0, 0, 0, err
		}

		lastTime := refTime
		if t, err := time.Parse(time.RFC3339Nano, accessedStr); err == nil {
			lastTime = t
		}
		if reinforcedStr.Valid {
			if t, err := time.Parse(time.RFC3339Nano, reinforcedStr.String); err == nil {
				if t.After(lastTime) {
					lastTime = t
				}
			}
		}

		elapsedSeconds := refTime.Sub(lastTime).Seconds()
		if elapsedSeconds < 0 {
			elapsedSeconds = 0
		}

		// Exponential decay formula: W(t) = W_0 * exp(-lambda * delta_t)
		decayFactor := math.Exp(-lambda * elapsedSeconds)
		decayedStability := stability * decayFactor

		shouldPrune := decayedStability < cfg.PruneThreshold && refTime.Sub(lastTime) >= cfg.InactivityGracePeriod

		decayQueue = append(decayQueue, nodeDecayItem{
			id:           id,
			newStability: decayedStability,
			shouldPrune:  shouldPrune,
		})
	}
	rows.Close()

	// 2. Batch commit node decay and soft-archival updates
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, 0, 0, err
	}
	defer func() { _ = tx.Rollback() }()

	stmtUpdate, err := tx.PrepareContext(ctx, `
		UPDATE nodes
		SET stability_score = ?
		WHERE id = ?
	`)
	if err != nil {
		return 0, 0, 0, err
	}
	defer stmtUpdate.Close()

	stmtArchive, err := tx.PrepareContext(ctx, `
		UPDATE nodes
		SET stability_score = ?,
		    is_archived = 1,
		    archived_at = ?
		WHERE id = ?
	`)
	if err != nil {
		return 0, 0, 0, err
	}
	defer stmtArchive.Close()

	refTimeStr := refTime.Format(time.RFC3339Nano)

	for _, item := range decayQueue {
		if item.shouldPrune {
			if _, err := stmtArchive.ExecContext(ctx, item.newStability, refTimeStr, item.id); err != nil {
				return 0, 0, 0, fmt.Errorf("failed archiving node %s: %w", item.id, err)
			}
			archived++
		} else {
			if _, err := stmtUpdate.ExecContext(ctx, item.newStability, item.id); err != nil {
				return 0, 0, 0, fmt.Errorf("failed updating decayed node %s: %w", item.id, err)
			}
			decayed++
		}
	}

	// 3. Prune weak edges and edges incident to soft-archived nodes
	edgeDeleteRes, err := tx.ExecContext(ctx, `
		DELETE FROM edges
		WHERE weight < ?
		   OR source_id IN (SELECT id FROM nodes WHERE is_archived = 1)
		   OR target_id IN (SELECT id FROM nodes WHERE is_archived = 1)
	`, cfg.EdgePruneThreshold)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed pruning decaying edges: %w", err)
	}

	if edgeDeleted, err := edgeDeleteRes.RowsAffected(); err == nil {
		prunedEdges = int(edgeDeleted)
	}

	if err := tx.Commit(); err != nil {
		return 0, 0, 0, fmt.Errorf("failed committing decay and prune transaction: %w", err)
	}

	return decayed, archived, prunedEdges, nil
}

// GetConsolidationStats calculates telemetry metrics for active vs soft-archived traces and graph growth.
func (s *SQLiteStore) GetConsolidationStats(ctx context.Context) (*model.ConsolidationStats, error) {
	var activeNodes, archivedNodes int64
	_ = s.db.QueryRowContext(ctx, "SELECT count(*) FROM nodes WHERE is_archived = 0").Scan(&activeNodes)
	_ = s.db.QueryRowContext(ctx, "SELECT count(*) FROM nodes WHERE is_archived = 1").Scan(&archivedNodes)

	var activeEdges int64
	_ = s.db.QueryRowContext(ctx, "SELECT count(*) FROM edges").Scan(&activeEdges)

	var tracesProcessed int64
	_ = s.db.QueryRowContext(ctx, "SELECT count(*) FROM episodic_traces WHERE consolidated = 1").Scan(&tracesProcessed)

	var meanStability, meanWeight float64
	_ = s.db.QueryRowContext(ctx, "SELECT coalesce(avg(stability_score), 0.0) FROM nodes WHERE is_archived = 0").Scan(&meanStability)
	_ = s.db.QueryRowContext(ctx, "SELECT coalesce(avg(weight), 0.0) FROM edges").Scan(&meanWeight)

	var dbSize int64
	if s.dbPath != "" && s.dbPath != ":memory:" {
		if fi, err := os.Stat(s.dbPath); err == nil {
			dbSize = fi.Size()
		}
	}

	return &model.ConsolidationStats{
		ActiveNodes:        activeNodes,
		ArchivedNodes:      archivedNodes,
		ActiveEdges:        activeEdges,
		TracesProcessed:    tracesProcessed,
		MeanStabilityScore: meanStability,
		MeanEdgeWeight:     meanWeight,
		DBSizeBytes:        dbSize,
	}, nil
}
