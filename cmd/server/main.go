package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/api"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/embedding"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/recall"
	"github.com/Duara-Cortex/sekha-knowledge-store/internal/store"
)

func main() {
	port := flag.Int("port", 8084, "Port for the HTTP knowledge graph service")
	dbPath := flag.String("db", "/var/lib/sekha/knowledge.db", "Path to persistent SQLite database file")
	decayTau := flag.Duration("tau", 24*time.Hour, "Recency decay half-life duration")
	alpha := flag.Float64("alpha", 0.6, "Default weight for semantic vector similarity")
	beta := flag.Float64("beta", 0.2, "Default weight for access count frequency")
	gamma := flag.Float64("gamma", 0.2, "Default weight for recency decay")
	flag.Parse()

	log.Printf("[Sekha Long-Term Memory] Initialising Knowledge Graph Store on Node 1...")
	log.Printf("[Config] Port: %d | DB: %s | Tau: %s | α: %.2f, β: %.2f, γ: %.2f",
		*port, *dbPath, *decayTau, *alpha, *beta, *gamma)

	// Ensure database parent directory exists
	dbDir := filepath.Dir(*dbPath)
	if err := os.MkdirAll(dbDir, 0755); err != nil {
		log.Printf("[Warning] Could not create db directory %s: %v. Falling back to local directory.", dbDir, err)
		*dbPath = "knowledge.db"
	}

	// Initialise SQLite store
	sqliteStore, err := store.NewSQLiteStore(*dbPath)
	if err != nil {
		log.Fatalf("[FATAL] Failed to initialise SQLite store at %s: %v", *dbPath, err)
	}
	defer sqliteStore.Close()

	embCfg := embedding.ConfigFromEnv()
	var embedClient embedding.Embedder
	if embCfg.Enabled {
		client := embedding.NewClient(embCfg, nil)
		embedClient = client
		log.Printf("[Embedding] Enabled: true | Endpoint: %s | Dims: %d | Status: %s",
			client.URL(), embCfg.Dimension, client.CheckHealth(context.Background()).Status)
	} else {
		log.Printf("[Embedding] Disabled (no embedding environment configured; operating without external embedding engine)")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	engineCfg := recall.EngineConfig{
		DefaultAlpha:       *alpha,
		DefaultBeta:        *beta,
		DefaultGamma:       *gamma,
		DefaultHybridAlpha: 0.65,
		DecayHalfLife:      *decayTau,
		FrequencyMaxCount:  100.0,
		NeighbourHopBoost:  0.35,
		MaxCandidatePool:   20,
		VectorDims:         embCfg.Dimension,
		Embedder:           embedClient,
	}

	engine, err := recall.NewEngine(ctx, sqliteStore, engineCfg)
	cancel()
	if err != nil {
		log.Fatalf("[FATAL] Failed to initialise associative recall engine: %v", err)
	}

	// Query initial counts for startup telemetry
	nCnt, eCnt, _ := sqliteStore.GetCounts(context.Background())
	log.Printf("[Status] Hydrated %d knowledge nodes and %d relational edges into in-memory index", nCnt, eCnt)

	// Launch periodic background synchronization with SQLite store
	syncCtx, cancelSync := context.WithCancel(context.Background())
	defer cancelSync()
	engine.StartBackgroundSync(syncCtx, 15*time.Second)

	server := api.NewServer(sqliteStore, engine, *port)

	httpServer := &http.Server{
		Addr:         fmt.Sprintf(":%d", *port),
		Handler:      server,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Graceful shutdown handling
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("[Sekha Knowledge Store] Listening on port %d", *port)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[FATAL] HTTP server failed: %v", err)
		}
	}()

	<-stop
	log.Printf("[Sekha Knowledge Store] Shutting down gracefully...")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()

	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("[Error] HTTP server shutdown error: %v", err)
	}

	log.Printf("[Sekha Knowledge Store] Offline.")
}
