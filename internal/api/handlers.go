package api

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/consolidation"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/recall"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

// Server implements the HTTP API for the knowledge graph store.
type Server struct {
	store               store.Store
	engine              *recall.Engine
	consolidationEngine *consolidation.Engine
	mux                 *http.ServeMux
	startTime           time.Time
	port                int
}

// NewServer builds and registers all API routes for port 8084.
func NewServer(s store.Store, e *recall.Engine, port int) *Server {
	srv := &Server{
		store:               s,
		engine:              e,
		consolidationEngine: consolidation.NewEngine(s, model.DefaultDecayConfig()),
		mux:                 http.NewServeMux(),
		startTime:           time.Now(),
		port:                port,
	}
	srv.registerRoutes()
	return srv
}

// SetConsolidationEngine overrides the default consolidation engine instance.
func (s *Server) SetConsolidationEngine(ce *consolidation.Engine) {
	if ce != nil {
		s.consolidationEngine = ce
	}
}

func (s *Server) registerRoutes() {
	s.mux.HandleFunc("POST /api/v1/memory/recall", s.handleRecall)
	s.mux.HandleFunc("POST /api/v1/memory/insert", s.handleInsert)
	s.mux.HandleFunc("POST /api/v1/memory/consolidate", s.handleConsolidate)
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
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	s.mux.ServeHTTP(w, r)
}

// handleRecall processes POST /api/v1/memory/recall.
func (s *Server) handleRecall(w http.ResponseWriter, r *http.Request) {
	var req model.RecallRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error": "invalid JSON payload in recall request: " + err.Error(),
		})
		return
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

	resp := model.HealthResponse{
		Status:        "healthy",
		Node:          "sekha-node1",
		Port:          s.port,
		Service:       "sekha-knowledge-store",
		UptimeSeconds: uptime,
		NodeCount:     nodeCount,
		EdgeCount:     edgeCount,
	}

	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
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
