package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/consolidation"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/embedding"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/recall"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

// Server implements the HTTP API for the knowledge graph store.
type Server struct {
	store               store.Store
	engine              *recall.Engine
	embedder            embedding.Embedder
	consolidationEngine *consolidation.Engine
	mux                 *http.ServeMux
	startTime           time.Time
	port                int
	apiKey              string
}

// NewServer builds and registers all API routes for port 8084.
func NewServer(s store.Store, e *recall.Engine, port int) *Server {
	ce := consolidation.NewEngine(s, model.DefaultDecayConfig())
	ce.AddNodeListener(func(nodes []model.Node) {
		e.RegisterNodes(nodes)
	})

	var emb embedding.Embedder
	if e != nil {
		emb = e.Embedder()
	}

	srv := &Server{
		store:               s,
		engine:              e,
		embedder:            emb,
		consolidationEngine: ce,
		mux:                 http.NewServeMux(),
		startTime:           time.Now(),
		port:                port,
		apiKey:              os.Getenv("SEKHA_API_KEY"),
	}
	srv.registerRoutes()
	return srv
}

// SetAPIKey configures the API key for endpoint authentication.
func (s *Server) SetAPIKey(key string) {
	s.apiKey = key
}

// APIKey returns the configured API key.
func (s *Server) APIKey() string {
	return s.apiKey
}

// SetEmbedder configures the dense embedding engine for the server.
func (s *Server) SetEmbedder(emb embedding.Embedder) {
	s.embedder = emb
	if s.engine != nil && emb != nil {
		s.engine.SetEmbedder(emb)
	}
}

// SetConsolidationEngine overrides the default consolidation engine instance.
func (s *Server) SetConsolidationEngine(ce *consolidation.Engine) {
	if ce != nil {
		s.consolidationEngine = ce
		s.consolidationEngine.AddNodeListener(func(nodes []model.Node) {
			s.engine.RegisterNodes(nodes)
		})
	}
}

func (s *Server) registerRoutes() {
	s.mux.HandleFunc("POST /api/v1/memory/recall", s.handleRecall)
	s.mux.HandleFunc("POST /api/v1/memory/insert", s.handleInsert)
	s.mux.HandleFunc("POST /api/v1/memory/consolidate", s.handleConsolidate)
	s.mux.HandleFunc("POST /api/v1/consolidation/decay", s.handleDecay)
	s.mux.HandleFunc("POST /api/v1/memory/decay", s.handleDecay)
	s.mux.HandleFunc("GET /api/v1/memory/consolidation/stats", s.handleConsolidationStats)
	s.mux.HandleFunc("GET /api/v1/memory/graph", s.handleGraph)
	s.mux.HandleFunc("GET /api/v1/memory/health", s.handleHealth)
	s.mux.HandleFunc("GET /health", s.handleHealth)
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Standard JSON Content-Type and CORS header
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, X-API-Key, Authorization")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if s.apiKey != "" {
		AuthMiddleware(s.apiKey, "/health", "/api/v1/memory/health")(s.mux).ServeHTTP(w, r)
		return
	}

	s.mux.ServeHTTP(w, r)
}

// handleRecall processes POST /api/v1/memory/recall.
func (s *Server) handleRecall(w http.ResponseWriter, r *http.Request) {
	var req model.RecallRequest
	if r.Body != nil {
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error": "failed reading request body: " + err.Error(),
			})
			return
		}
		if len(bytes.TrimSpace(bodyBytes)) > 0 {
			if err := json.Unmarshal(bodyBytes, &req); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				_ = json.NewEncoder(w).Encode(map[string]string{
					"error": "invalid JSON payload in recall request: " + err.Error(),
				})
				return
			}
		}
	}

	// Parse URL query parameters to support and/or override request parameters
	q := r.URL.Query()
	if q.Has("include_embeddings") {
		val := strings.ToLower(strings.TrimSpace(q.Get("include_embeddings")))
		req.IncludeEmbeddings = (val == "true" || val == "1" || val == "yes")
	}
	if q.Has("include_secrets") {
		val := strings.ToLower(strings.TrimSpace(q.Get("include_secrets")))
		req.IncludeSecrets = (val == "true" || val == "1" || val == "yes")
	}
	if q.Has("hybrid_alpha") {
		if val, err := strconv.ParseFloat(strings.TrimSpace(q.Get("hybrid_alpha")), 64); err == nil {
			req.HybridAlpha = &val
		}
	}
	if q.Has("mode") {
		req.Mode = strings.TrimSpace(q.Get("mode"))
	}
	if q.Has("fields") {
		var queryFields []string
		for _, fVal := range q["fields"] {
			for _, f := range strings.Split(fVal, ",") {
				clean := strings.TrimSpace(f)
				if clean != "" {
					queryFields = append(queryFields, clean)
				}
			}
		}
		if len(queryFields) > 0 {
			req.Fields = queryFields
		}
	}
	if q.Has("expand_neighbours") {
		val := strings.ToLower(strings.TrimSpace(q.Get("expand_neighbours")))
		b := (val == "true" || val == "1" || val == "yes")
		req.ExpandNeighbours = &b
	}
	if q.Has("expand_hops") {
		if val, err := strconv.Atoi(strings.TrimSpace(q.Get("expand_hops"))); err == nil {
			req.ExpandHops = val
		}
	}
	if q.Has("min_edge_weight") {
		if val, err := strconv.ParseFloat(strings.TrimSpace(q.Get("min_edge_weight")), 64); err == nil {
			req.MinEdgeWeight = &val
		}
	}
	if q.Has("traverse_relations") {
		var rels []string
		for _, rVal := range q["traverse_relations"] {
			for _, rItem := range strings.Split(rVal, ",") {
				clean := strings.TrimSpace(rItem)
				if clean != "" {
					rels = append(rels, clean)
				}
			}
		}
		if len(rels) > 0 {
			req.TraverseRelations = rels
		}
	}
	if q.Has("attenuation_factor") {
		if val, err := strconv.ParseFloat(strings.TrimSpace(q.Get("attenuation_factor")), 64); err == nil {
			req.AttenuationFactor = &val
		}
	}

	resp, err := s.engine.Recall(r.Context(), req)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error": "recall failed: " + err.Error(),
		})
		return
	}

	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

// handleInsert processes POST /api/v1/memory/insert.
func (s *Server) handleInsert(w http.ResponseWriter, r *http.Request) {
	var req model.InsertRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(model.InsertResponse{
			Status: "error",
			Error:  "invalid JSON payload: " + err.Error(),
		})
		return
	}

	// Compute 384-D dense embedding on-the-fly if vector is omitted in the insert payload
	for i := range req.Nodes {
		if len(req.Nodes[i].Embedding) == 0 {
			text := strings.TrimSpace(req.Nodes[i].Label + " " + req.Nodes[i].Summary)
			if text != "" {
				if s.embedder != nil {
					if emb, err := s.embedder.EmbedText(r.Context(), text); err == nil && len(emb) > 0 {
						req.Nodes[i].Embedding = emb
					}
				}
				if len(req.Nodes[i].Embedding) == 0 {
					req.Nodes[i].Embedding = embedding.Generate(text, model.DefaultVectorDim)
				}
			}
		}
	}

	insertedNodes, err := s.store.InsertNodes(r.Context(), req.Nodes)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(model.InsertResponse{
			Status: "error",
			Error:  "node insertion failed: " + err.Error(),
		})
		return
	}

	insertedEdges, err := s.store.InsertEdges(r.Context(), req.Edges)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(model.InsertResponse{
			Status:        "partial_error",
			InsertedNodes: insertedNodes,
			Error:         "edge insertion failed: " + err.Error(),
		})
		return
	}

	// Register inserted nodes with in-memory vector index
	if len(req.Nodes) > 0 {
		s.engine.RegisterNodes(req.Nodes)
	}

	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(model.InsertResponse{
		Status:        "success",
		InsertedNodes: insertedNodes,
		InsertedEdges: insertedEdges,
	})
}

// handleGraph processes GET /api/v1/memory/graph.
func (s *Server) handleGraph(w http.ResponseWriter, r *http.Request) {
	summary, err := s.store.GetGraphSummary(r.Context())
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error": "failed calculating graph summary: " + err.Error(),
		})
		return
	}

	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(summary)
}

// handleHealth processes GET /api/v1/memory/health.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	nodeCount, edgeCount, _ := s.store.GetCounts(r.Context())
	uptime := int64(time.Since(s.startTime).Seconds())

	embHealth := s.getEmbeddingHealth(r.Context())

	resp := model.HealthResponse{
		Status:          "healthy",
		Node:            "sekha-node1",
		Port:            s.port,
		Service:         "sekha-knowledge-store",
		UptimeSeconds:   uptime,
		NodeCount:       nodeCount,
		EdgeCount:       edgeCount,
		EmbeddingEngine: &embHealth,
	}

	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) getEmbeddingHealth(ctx context.Context) model.EmbeddingEngineHealth {
	if s.embedder != nil {
		if client, ok := s.embedder.(*embedding.Client); ok {
			return client.CheckHealth(ctx)
		}
		if mock, ok := s.embedder.(*embedding.MockEmbedder); ok {
			url := os.Getenv("SEKHA_EMBEDDING_URL")
			if url == "" {
				url = "mock"
			}
			return model.EmbeddingEngineHealth{
				Enabled:   true,
				Status:    "reachable",
				URL:       url,
				Dimension: mock.Dimension(),
			}
		}
		return model.EmbeddingEngineHealth{
			Enabled:   true,
			Status:    "reachable",
			URL:       os.Getenv("SEKHA_EMBEDDING_URL"),
			Dimension: model.DefaultVectorDim,
		}
	}

	return model.EmbeddingEngineHealth{
		Enabled:   false,
		Status:    "offline",
		URL:       os.Getenv("SEKHA_EMBEDDING_URL"),
		Dimension: 0,
	}
}

// handleConsolidate processes POST /api/v1/memory/consolidate dispatched from Node 2.
func (s *Server) handleConsolidate(w http.ResponseWriter, r *http.Request) {
	var req model.ConsolidateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(model.ConsolidateResponse{
			Status:  "error",
			Message: "invalid JSON payload: " + err.Error(),
		})
		return
	}

	resp, err := s.consolidationEngine.IngestTrace(r.Context(), req)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(model.ConsolidateResponse{
			Status:  "error",
			Message: "trace consolidation failed: " + err.Error(),
		})
		return
	}

	// Compute dense embeddings on-the-fly for any created nodes missing vectors
	if len(resp.CreatedNodes) > 0 {
		for i := range resp.CreatedNodes {
			if len(resp.CreatedNodes[i].Embedding) == 0 {
				text := strings.TrimSpace(resp.CreatedNodes[i].Label + " " + resp.CreatedNodes[i].Summary)
				if text != "" {
					if s.embedder != nil {
						if emb, err := s.embedder.EmbedText(r.Context(), text); err == nil && len(emb) > 0 {
							resp.CreatedNodes[i].Embedding = emb
						}
					}
					if len(resp.CreatedNodes[i].Embedding) == 0 {
						resp.CreatedNodes[i].Embedding = embedding.Generate(text, model.DefaultVectorDim)
					}
				}
			}
		}
		// Register novel consolidated nodes directly with in-memory recall index
		s.engine.RegisterNodes(resp.CreatedNodes)
	}

	if resp.Synchronous {
		w.WriteHeader(http.StatusOK)
	} else {
		w.WriteHeader(http.StatusAccepted)
	}
	_ = json.NewEncoder(w).Encode(resp)
}

// handleConsolidationStats processes GET /api/v1/memory/consolidation/stats.
func (s *Server) handleConsolidationStats(w http.ResponseWriter, r *http.Request) {
	stats, err := s.consolidationEngine.GetStats(r.Context())
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error": "failed fetching consolidation stats: " + err.Error(),
		})
		return
	}

	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(stats)
}

// handleDecay processes POST /api/v1/consolidation/decay and POST /api/v1/memory/decay.
func (s *Server) handleDecay(w http.ResponseWriter, r *http.Request) {
	var req model.DecayRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error": "invalid JSON payload: " + err.Error(),
		})
		return
	}

	res, err := s.consolidationEngine.ExecuteScopedDecay(r.Context(), req, time.Now().UTC())
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error": "scoped decay failed: " + err.Error(),
		})
		return
	}

	// Rehydrate in-memory recall engine if nodes were soft-archived
	if res.NodesArchived > 0 && s.engine != nil {
		_ = s.engine.Hydrate(r.Context())
	}

	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(res)
}
