# 🧠 sekha-knowledge-store

**Persistent Relational Knowledge Graph, Associative Retrieval & Memory Consolidation Subsystem for Node 1**  
*Part of the Sekha Tri-Node Edge Cognitive Cluster &bull; Duara Cortex Q3 2026*

`sekha-knowledge-store` is a local-first, low-latency persistent knowledge graph and memory consolidation subsystem engineered in pure Go (with zero CGO dependencies) running on **Node 1** (`sekha-node1` &bull; `192.168.8.213` &bull; 8GB RAM). It acts as the Long-Term Memory (LTM) subsystem of the Sekha cognitive cluster, executing low-latency associative recall alongside background memory consolidation and mathematical recency decay.

---

## Key Features

1. **Local-First SQLite Engine (Zero CGO):**
   - Pure Go SQLite runtime (`modernc.org/sqlite`) compatible with static compilation (`CGO_ENABLED=0`) across ARM64 Linux and macOS.
   - WAL journal mode, in-memory temp store, and 64MB page cache for high-concurrency read throughput and non-blocking background consolidation.
2. **Associative Recall Scoring:**
   - Multi-factor retrieval scoring combining semantic similarity, usage frequency, and recency:
     $$R(\text{node}) = \alpha \cdot \text{Sim}(q, \text{embedding}) + \beta \cdot \log(1 + \text{access\_count}) + \gamma \cdot \text{recency\_decay}$$
   - Recency decay formulated as $e^{-\Delta t / \tau}$ with configurable half-life $\tau$.
3. **Hybrid Search + 1-Hop Graph Expansion:**
   - Evaluates vector embeddings against seed nodes, then traverses the relational `edges` table to boost connected 1-hop neighbours.
   - Returns ranked contextual nodes alongside interconnecting subgraph edges.
4. **Memory Consolidation & Graph Fusion Daemon (`sekha-consolidation`):**
   - Ingests completed episodic deliberation traces dispatched from Node 2 working memory (`POST /api/v1/memory/consolidate`).
   - Extracts salient concepts, decisions, and causal relationships (`subgoal_of`, `precedes`, `associates_with`, `context_for`).
   - Performs graph fusion and entity deduplication against persistent SQLite knowledge vertices.
5. **Mathematical Decay & Hebbian Reinforcement Engine:**
   - Exponential recency decay for unreinforced nodes:
     $$W(t) = W_0 \cdot \exp(-\lambda (t - t_{\text{last}})) \quad \text{where } \lambda = \frac{\ln(2)}{\tau_{\text{decay}}}$$
   - Hebbian co-activation reinforcement for edges during successful task completions:
     $$W_{\text{new}} = \min(W_{\text{max}},\, W_{\text{old}} + \Delta W)$$
   - Soft-archives decaying nodes whose stability score drops below threshold $\omega_{\text{prune}} = 0.10$ after an inactivity grace period, pruning weak edges below $0.05$ to prevent unbounded database bloat.
6. **Sub-20ms Retrieval & High-Throughput Batch Cycles:**
   - In-memory vector index cache ensures recall operations across 10,000 nodes execute in `< 2.0ms`.
   - Periodic 15-minute background batch cycle fuses queued traces and prunes decaying state without blocking query threads.
7. **Cluster Telemetry & Observability:**
   - Integrated with Node 1's native `/usr/local/bin/sekha` CLI dashboard for temperature, memory, service status, and topology health checks.

---

## Relational Schema

### `nodes` Table
```sql
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
```

### `edges` Table
```sql
CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP NOT NULL,
    last_reinforced_at TIMESTAMP,
    PRIMARY KEY (source_id, target_id, relation_type)
);
```

### `episodic_traces` Table
```sql
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
```

---

## REST API Endpoints

### Knowledge Graph Store (`sekha-knowledge-store` &bull; Port 8084)

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/memory/recall` | Associative recall query with vector embedding or entity ID and 1-hop expansion |
| `POST` | `/api/v1/memory/insert` | Commits persistent knowledge nodes and edges with upsert logic |
| `POST` | `/api/v1/memory/consolidate` | Ingests resolved episodic deliberation traces from Node 2 working memory |
| `GET` | `/api/v1/memory/consolidation/stats` | Returns consolidation retention, archival, and graph growth metrics |
| `GET` | `/api/v1/memory/graph` | Returns topological metrics (node count, edge count, density, average degree) |
| `GET` | `/api/v1/memory/health` | Health telemetry and uptime status |

### Memory Consolidation Daemon (`sekha-consolidation` &bull; Port 8085)

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/consolidation/trigger` | Triggers an immediate out-of-band consolidation, decay, and pruning cycle |
| `POST` | `/api/v1/memory/consolidate` | Direct trace ingestion endpoint with optional synchronous consolidation |
| `GET` | `/api/v1/consolidation/stats` | Telemetry on active vs archived nodes, reinforced edges, and mean stability |
| `GET` | `/api/v1/consolidation/health` | Subsystem health telemetry |

---

## Building & Installation

### On Development Machine:
```bash
# Build native binaries
make build

# Cross-compile for Node 1 (ARM64 Raspberry Pi 5)
make build-arm64

# Run 10,000-node empirical recall benchmark
make validate

# Run 1,000-event consolidation and decay stress test
make validate-consolidation
```

### On Node 1 (`sekha-node1` &bull; `192.168.8.213`):
```bash
# Clone repository
git clone git@github.com:Duara-Cortex/sekha-knowledge-store.git
cd sekha-knowledge-store

# Compile and register both systemd daemons
make build
sudo make install

# Check status via native CLI
sekha status
```
