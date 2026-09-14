# 🧠 sekha-knowledge-store

**Persistent Relational Knowledge Graph & Associative Retrieval Service for Node 1**  
*Part of the Sekha Tri-Node Edge Cognitive Cluster &bull; Duara Cortex Q3 2026*

`sekha-knowledge-store` is a local-first, low-latency persistent knowledge graph daemon engineered in pure Go (with zero CGO dependencies) running on **Node 1** (`sekha-node1` &bull; `192.168.8.213` &bull; 8GB RAM). It acts as the digital Long-Term Memory (LTM) subsystem of the Sekha cognitive cluster.

---

## Key Features

1. **Local-First SQLite Engine (Zero CGO):**
   - Pure Go SQLite runtime (`modernc.org/sqlite`) compatible with static compilation (`CGO_ENABLED=0`) across ARM64 Linux and macOS.
   - WAL journal mode, in-memory temp store, and 64MB page cache for high-concurrency read throughput.
2. **Associative Recall Scoring:**
   - Multi-factor retrieval scoring combining semantic similarity, usage frequency, and recency:
     $$R(\text{node}) = \alpha \cdot \text{Sim}(q, \text{embedding}) + \beta \cdot \log(1 + \text{access\_count}) + \gamma \cdot \text{recency\_decay}$$
   - Recency decay formulated as $e^{-\Delta t / \tau}$ with configurable half-life $\tau$.
3. **Hybrid Search + 1-Hop Graph Expansion:**
   - Evaluates vector embeddings against seed nodes, then traverses the relational `edges` table to boost connected 1-hop neighbours.
   - Returns ranked contextual nodes alongside interconnecting subgraph edges.
4. **Sub-20ms Retrieval Latency:**
   - In-memory vector index cache ensures recall operations across 10,000 nodes execute in `< 2.0ms` on stock Raspberry Pi 5 hardware.
5. **Cluster Telemetry & Observability:**
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
    access_count INTEGER NOT NULL DEFAULT 0,
    stability_score REAL NOT NULL DEFAULT 1.0
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
    PRIMARY KEY (source_id, target_id, relation_type)
);
```

---

## REST API Endpoints (Port 8084)

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/memory/recall` | Associative recall query with vector embedding or entity ID and 1-hop expansion |
| `POST` | `/api/v1/memory/insert` | Commits persistent knowledge nodes and edges with upsert logic |
| `GET` | `/api/v1/memory/graph` | Returns topological metrics (node count, edge count, density, average degree) |
| `GET` | `/api/v1/memory/health` | Health telemetry and uptime status |

---

## Building & Installation

### On Development Machine:
```bash
# Build native binaries
make build

# Cross-compile for Node 1 (ARM64 Raspberry Pi 5)
make build-arm64

# Run 10,000-node empirical benchmark
make benchmark
```

### On Node 1 (`sekha-node1` • `192.168.8.213`):
```bash
# Clone repository
git clone git@github.com:Duara-Cortex/sekha-knowledge-store.git
cd sekha-knowledge-store

# Compile and register systemd daemon
make build
sudo make install

# Check status via native CLI
sekha status
```
