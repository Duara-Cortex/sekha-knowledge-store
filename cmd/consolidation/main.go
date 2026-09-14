package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/consolidation"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

func main() {
	port := flag.Int("port", 8085, "Port for the consolidation daemon telemetry and trigger service")
	dbPath := flag.String("db", "/var/lib/sekha/knowledge.db", "Path to persistent SQLite database file")
	cycleInterval := flag.Duration("cycle", 15*time.Minute, "Periodic background consolidation cycle interval")
	decayTau := flag.Duration("decay-tau", 72*time.Hour, "Recency decay half-life duration (tau)")
	pruneThreshold := flag.Float64("prune-threshold", 0.10, "Retention threshold omega_prune below which nodes are soft-archived")
	edgePruneThreshold := flag.Float64("edge-prune-threshold", 0.05, "Minimum edge weight before pruning")
	gracePeriod := flag.Duration("grace-period", 168*time.Hour, "Inactivity window before unreinforced node is soft-archived")
	hebbianRate := flag.Float64("hebbian-rate", 0.15, "Hebbian learning rate eta for co-activation reinforcement")
	maxWeight := flag.Float64("max-weight", 5.0, "Saturation upper bound for edge weight reinforcement")
	batchSize := flag.Int("batch-size", 500, "Maximum batch size for SQLite transactions")
	flag.Parse()

	log.Println("=================================================================")
	log.Println("   Sekha Memory Consolidation & Mathematical Decay Daemon        ")
	log.Println("   Node 1 (8GB RAM Pi 5) Long-Term Memory Subsystem              ")
	log.Println("=================================================================")
	log.Printf("[Config] Port: %d | DB: %s | Cycle: %s | Tau: %s", *port, *dbPath, *cycleInterval, *decayTau)
	log.Printf("[Config] Prune Threshold (omega): %.2f | Edge Prune: %.2f | Grace Period: %s",
		*pruneThreshold, *edgePruneThreshold, *gracePeriod)
	log.Printf("[Config] Hebbian Learning Rate (eta): %.2f | Max Edge Weight: %.2f | Batch Size: %d",
		*hebbianRate, *maxWeight, *batchSize)

	// Ensure database parent directory exists
	dbDir := filepath.Dir(*dbPath)
	if err := os.MkdirAll(dbDir, 0755); err != nil {
		log.Printf("[Warning] Could not create database directory %s: %v. Using local directory.", dbDir, err)
		*dbPath = "knowledge.db"
	}

	// Initialise SQLite store
	sqliteStore, err := store.NewSQLiteStore(*dbPath)
	if err != nil {
		log.Fatalf("[FATAL] Failed to initialise SQLite store at %s: %v", *dbPath, err)
	}
	defer sqliteStore.Close()

	decayCfg := model.DecayConfig{
		DecayHalfLife:         *decayTau,
		PruneThreshold:        *pruneThreshold,
		EdgePruneThreshold:    *edgePruneThreshold,
		InactivityGracePeriod: *gracePeriod,
		HebbianLearningRate:   *hebbianRate,
		MaxEdgeWeight:         *maxWeight,
		BatchSize:             *batchSize,
	}

	engine := consolidation.NewEngine(sqliteStore, decayCfg)

	// Context for scheduler lifecycle
	schedulerCtx, cancelScheduler := context.WithCancel(context.Background())
	defer cancelScheduler()

	// Launch background consolidation scheduler
	go engine.StartScheduler(schedulerCtx, *cycleInterval)

	// HTTP management, trigger, and telemetry server
	mux := http.NewServeMux()
	startTime := time.Now()

	mux.HandleFunc("POST /api/v1/consolidation/trigger", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		res, err := engine.RunConsolidationCycle(r.Context(), time.Now().UTC())
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(res)
	})

	mux.HandleFunc("POST /api/v1/memory/consolidate", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		var req model.ConsolidateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(model.ConsolidateResponse{
				Status:  "error",
				Message: "invalid JSON payload: " + err.Error(),
			})
			return
		}
		res, err := engine.IngestTrace(r.Context(), req)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(model.ConsolidateResponse{
				Status:  "error",
				Message: err.Error(),
			})
			return
		}
		if res.Synchronous {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusAccepted)
		}
		_ = json.NewEncoder(w).Encode(res)
	})

	mux.HandleFunc("GET /api/v1/consolidation/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		stats, err := engine.GetStats(r.Context())
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(stats)
	})

	mux.HandleFunc("GET /api/v1/consolidation/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		stats, _ := engine.GetStats(r.Context())
		uptime := int64(time.Since(startTime).Seconds())
		resp := map[string]interface{}{
			"status":         "healthy",
			"node":           "sekha-node1",
			"port":           *port,
			"service":        "sekha-consolidation",
			"uptime_seconds": uptime,
			"total_cycles":   stats.TotalCycles,
			"active_nodes":   stats.ActiveNodes,
			"archived_nodes": stats.ArchivedNodes,
			"active_edges":   stats.ActiveEdges,
		}
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(resp)
	})

	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "healthy", "service": "sekha-consolidation"})
	})

	httpServer := &http.Server{
		Addr:         fmt.Sprintf(":%d", *port),
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("[Sekha Consolidation Daemon] Listening on http://0.0.0.0:%d", *port)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[FATAL] HTTP server encountered failure: %v", err)
		}
	}()

	<-stop
	log.Println("[Sekha Consolidation Daemon] Received termination signal; initiating graceful shutdown...")

	cancelScheduler()

	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancelShutdown()

	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("[Error] HTTP server shutdown error: %v", err)
	}

	log.Println("[Sekha Consolidation Daemon] Offline.")
}
