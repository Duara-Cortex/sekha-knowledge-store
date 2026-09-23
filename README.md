# 🧠 sekha-knowledge-store

**Persistent Relational Knowledge Graph, Associative Retrieval & Memory Consolidation Subsystem for Node 1**  
*Part of the Sekha Tri-Node Edge Cognitive Cluster &bull; Duara Cortex Q3 2026*

`sekha-knowledge-store` is a local-first, low-latency persistent knowledge graph and memory consolidation subsystem engineered in pure Go (with zero CGO dependencies) running on **Node 1** (`sekha-node1` &bull; 8GB RAM ARM64). It acts as the Long-Term Memory (LTM) subsystem of the Sekha cognitive cluster, executing low-latency associative recall alongside background memory consolidation and mathematical recency decay.

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
| `GET` | `/api/v1/consolidation/health` | Subsystem health telemetry (public unauthenticated) |

---

## Security & Endpoint Hardening

`sekha-knowledge-store` and `sekha-consolidation` provide defense-in-depth protections for edge and cluster deployments:

1. **API Key & Bearer Token Authentication:**
   - When `SEKHA_API_KEY` (or `--api-key`) is configured, all mutation and recall endpoints (`/recall`, `/insert`, `/consolidate`, `/decay`, `/trigger`, etc.) require authentication via:
     - `X-API-Key: <key>`
     - `Authorization: Bearer <key>`
   - Requests without a valid key receive `401 Unauthorized` (`{"error": "unauthorized: invalid or missing API key"}`).
   - **Public Exemption:** Telemetry and health check endpoints (`/health`, `/api/v1/memory/health`, `/api/v1/consolidation/health`) remain accessible without credentials so node monitoring daemons (such as `/usr/local/bin/sekha status`) function uninterrupted.
   - If `SEKHA_API_KEY` is empty or unset, the daemons operate in permissive development mode with an explicit startup warning.

2. **TLS / HTTPS Transport:**
   - Both daemons support encrypted TLS transport via `--tls-cert` and `--tls-key` flags or `SEKHA_TLS_CERT` and `SEKHA_TLS_KEY` environment variables.
   - When both certificate and key are provided, servers listen over HTTPS with TLS 1.2/1.3.

3. **AES-256-GCM Encryption-at-Rest (`is_secret`):**
   - The relational schema includes an `is_secret INTEGER NOT NULL DEFAULT 0` column on the `nodes` table.
   - When `is_secret: true` is passed during node insertion:
     - The node summary/payload is encrypted using AES-256-GCM with a 12-byte cryptographically secure random nonce.
     - Ciphertext is persisted with standard prefix format: `enc:v1:<base64(12-byte nonce + ciphertext + 16-byte tag)>`.
     - The encryption key is derived via SHA-256 from `SEKHA_MASTER_KEY` (or `/etc/sekha/master.key` with `0600` permissions).
     - Plaintext secrets never appear in the SQLite database file or WAL logs.
   - **Vault / Secret Manager References:** Values starting with `vault://` or `env://` are recognized as external URI references and preserved without double-encryption.

4. **Secret Masking & Redaction on Recall:**
   - By default, secret nodes retrieved during associative recall have their summaries masked as `[REDACTED_SECRET]` to prevent accidental prompt injection or leakage to unauthorized agents.
   - When `include_secrets: true` is provided in the JSON body (or `?include_secrets=true` query parameter) of an authenticated recall request, the payload is decrypted and returned in plaintext.


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

### On Target Node:
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

---

## Configuration & Environment

The knowledge store supports optional dense semantic embeddings combined with in-memory lexical BM25 search. Configuration is loaded from environment variables, `/etc/default/sekha` (systemd on Linux), or a local `.env` file (Windows / macOS).

### Hardware Considerations: Edge Defaults vs. High-Performance Hardware

`sekha-knowledge-store` was engineered and benchmarked for resource-constrained edge hardware (such as a **Raspberry Pi 5 ARM64 with 8GB RAM**):
- **Edge Defaults:** Tuned for low memory and CPU constraints. By default, it uses compact 384-D dense embeddings (`all-MiniLM-L6-v2`) taking only ~15 MB of RAM per 10,000 nodes, achieving sub-15ms associative recall purely on CPU without GPU acceleration.
- **Scaling for High-Performance Hardware:** If you are running on more capable hardware (multi-core desktop CPUs, high-RAM workstations, or servers with dedicated GPUs):
  - **Higher-Dimensional Embeddings:** You are not restricted to 384-D vectors. You can plug in higher-dimensional models (such as 768-D or 1024-D embeddings like `bge-large`, `nomic-embed-text`, or `text-embedding-3-small`) simply by setting `SEKHA_EMBEDDING_DIM` and updating `SEKHA_EMBEDDING_URL`.
  - **Large-Scale Knowledge Graphs:** Memory consumption scales linearly (~1.5 KB per 384-D node). A workstation or server with 16GB–64GB RAM can host millions of nodes entirely in-memory with sub-10ms recall.
  - **Sub-Millisecond Query Latencies:** Systems with wide SIMD vector pipelines (AVX2, AVX-512, or Apple Silicon NEON) execute linear dot product scans in `< 1.0ms`.
  - **Adjustable Timeouts:** You can tune `SEKHA_EMBEDDING_TIMEOUT_MS` based on whether your embedding model runs locally on a fast GPU or over a network.

### Linux Deployment (`/etc/default/sekha`)

On Linux/systemd nodes, create or edit `/etc/default/sekha` directly:

```bash
sudo nano /etc/default/sekha
```

Add your environment configuration:

```bash
# =====================================================================
# Sekha Knowledge Store Configuration
# Location: /etc/default/sekha (or .env in project root)
# =====================================================================

# Enable the dense embedding engine (set to true to connect to service)
SEKHA_EMBEDDING_ENABLED=true

# Full URL to your OpenAI-compatible /v1/embeddings endpoint
# Replace <embedding-host> and <port> with your embedding service host/IP and configured port
SEKHA_EMBEDDING_URL=http://<embedding-host>:<port>/v1/embeddings

# Vector dimension (384 for all-MiniLM-L6-v2, or higher if using larger models)
SEKHA_EMBEDDING_DIM=384

# Request timeout in milliseconds
SEKHA_EMBEDDING_TIMEOUT_MS=500

# Optional API key for embedding service on Port 8086 (falls back to SEKHA_API_KEY if unset)
# SEKHA_EMBEDDING_API_KEY=your_embedding_service_key_here

# Security & Authentication Configuration
# Set an API key to enforce authentication across non-health endpoints (pass via X-API-Key or Bearer token)
SEKHA_API_KEY=your_cluster_secret_key_here

# Node-local master key for AES-256-GCM encryption-at-rest (or stored in /etc/sekha/master.key)
SEKHA_MASTER_KEY=your_secure_random_node_master_key_here

# TLS / HTTPS transport configuration (optional)
# SEKHA_TLS_CERT=/etc/ssl/certs/sekha.crt
# SEKHA_TLS_KEY=/etc/ssl/private/sekha.key
```

After modifying `/etc/default/sekha`, restart the services to apply changes:
```bash
sudo systemctl restart sekha-knowledge-store sekha-consolidation
```

### Windows & macOS Environments (`.env`)

Because Windows does not have an `/etc` directory and does not use `systemd`, **Windows users must use a `.env` file** in the project root instead of `/etc/default/sekha`.

1. Create a `.env` file in the root of the repository:
   ```bash
   SEKHA_EMBEDDING_ENABLED=true
   SEKHA_EMBEDDING_URL=http://<embedding-host>:<port>/v1/embeddings
   SEKHA_EMBEDDING_DIM=384
   SEKHA_EMBEDDING_TIMEOUT_MS=500
   # SEKHA_EMBEDDING_API_KEY=your_embedding_service_key_here

   # Security & Authentication
   SEKHA_API_KEY=your_cluster_secret_key_here
   SEKHA_MASTER_KEY=your_secure_random_node_master_key_here
   # SEKHA_TLS_CERT=/path/to/sekha.crt
   # SEKHA_TLS_KEY=/path/to/sekha.key
   ```

2. The application automatically detects and loads `.env` upon startup on Windows and macOS:
   ```powershell
   # PowerShell (Windows)
   .\bin\sekha-knowledge-store.exe -port <port> -db knowledge.db
   ```
   Or set environment variables directly in your shell session:
   ```powershell
   $env:SEKHA_EMBEDDING_ENABLED="true"
   $env:SEKHA_EMBEDDING_URL="http://<embedding-host>:<port>/v1/embeddings"
   ```

### Standalone Mode (Zero External Dependencies)

If no environment configuration exists (or if `SEKHA_EMBEDDING_ENABLED=false`), the subsystem automatically sets itself up in **standalone mode**:
- Operates with pure in-memory lexical BM25 search and local deterministic projections.
- Requires no external embedding service or network connectivity.
- `make install` will never force-create `/etc/default/sekha`, and will preserve your existing file if one is already present on the node.




