package embedding

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Duara-Cortex/sekha-knowledge-store/internal/model"
)

// Embedder defines the contract for generating dense semantic vector embeddings.
type Embedder interface {
	EmbedText(ctx context.Context, text string) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}

// Config holds configuration parameters for the edge-local embedding service.
type Config struct {
	Enabled   bool          // SEKHA_EMBEDDING_ENABLED (default: false unless explicitly enabled)
	URL       string        // SEKHA_EMBEDDING_URL
	Dimension int           // SEKHA_EMBEDDING_DIM (default: 384)
	Timeout   time.Duration // SEKHA_EMBEDDING_TIMEOUT_MS (default: 500ms)
	APIKey    string        // SEKHA_EMBEDDING_API_KEY (fallback to SEKHA_API_KEY)
}

// DefaultConfig returns the default configuration when embedding is disabled.
func DefaultConfig() Config {
	return Config{
		Enabled:   false,
		URL:       "",
		Dimension: model.DefaultVectorDim, // 384
		Timeout:   500 * time.Millisecond,
		APIKey:    "",
	}
}

// LoadEnvFile reads key=value pairs from a file and sets them in the environment
// if they are not already set.
func LoadEnvFile(path string) error {
	if strings.TrimSpace(path) == "" {
		return nil
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			k := strings.TrimSpace(parts[0])
			v := strings.TrimSpace(parts[1])
			v = strings.Trim(v, `"'`)
			if os.Getenv(k) == "" {
				_ = os.Setenv(k, v)
			}
		}
	}
	return scanner.Err()
}

// ConfigFromEnv loads embedding configuration from environment variables or env files.
func ConfigFromEnv() Config {
	// Attempt to load from env file if present
	if envPath := os.Getenv("SEKHA_ENV_FILE"); envPath != "" {
		_ = LoadEnvFile(envPath)
	} else {
		if _, err := os.Stat("/etc/default/sekha"); err == nil {
			_ = LoadEnvFile("/etc/default/sekha")
		} else if _, err := os.Stat(".env"); err == nil {
			_ = LoadEnvFile(".env")
		}
	}

	cfg := DefaultConfig()

	if val := os.Getenv("SEKHA_EMBEDDING_ENABLED"); val != "" {
		lower := strings.ToLower(strings.TrimSpace(val))
		cfg.Enabled = (lower == "true" || lower == "1" || lower == "yes" || lower == "t")
	}

	url := strings.TrimSpace(os.Getenv("SEKHA_EMBEDDING_URL"))
	if url == "" {
		host := strings.TrimSpace(os.Getenv("SEKHA_EMBEDDING_HOST"))
		if host != "" {
			port := strings.TrimSpace(os.Getenv("SEKHA_EMBEDDING_PORT"))
			if port == "" {
				port = "8086"
			}
			url = fmt.Sprintf("http://%s:%s/v1/embeddings", host, port)
		}
	}
	if url != "" {
		cfg.URL = url
	} else if cfg.Enabled {
		// If explicitly enabled but no URL or host provided, check SEKHA_HOST or fallback to localhost
		host := strings.TrimSpace(os.Getenv("SEKHA_HOST"))
		if host == "" {
			host = "localhost"
		}
		cfg.URL = fmt.Sprintf("http://%s:8086/v1/embeddings", host)
	}

	if val := os.Getenv("SEKHA_EMBEDDING_DIM"); val != "" {
		if dim, err := strconv.Atoi(strings.TrimSpace(val)); err == nil && dim > 0 {
			cfg.Dimension = dim
		}
	}

	if val := os.Getenv("SEKHA_EMBEDDING_TIMEOUT_MS"); val != "" {
		if ms, err := strconv.Atoi(strings.TrimSpace(val)); err == nil && ms > 0 {
			cfg.Timeout = time.Duration(ms) * time.Millisecond
		}
	}

	apiKey := strings.TrimSpace(os.Getenv("SEKHA_EMBEDDING_API_KEY"))
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("SEKHA_API_KEY"))
	}
	cfg.APIKey = apiKey

	return cfg
}

// openAIEmbeddingRequest is the JSON payload sent to standard /v1/embeddings endpoints.
type openAIEmbeddingRequest struct {
	Input any `json:"input"` // string or []string
}

// openAIEmbeddingItem holds an individual vector embedding in an OpenAI-compatible response.
type openAIEmbeddingItem struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// openAIEmbeddingResponse models the response from standard /v1/embeddings endpoints.
type openAIEmbeddingResponse struct {
	Object string                `json:"object"`
	Data   []openAIEmbeddingItem `json:"data"`
	Error  *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Client connects to the edge-local embedding service with automatic fallback to MockEmbedder.
type Client struct {
	cfg        Config
	httpClient *http.Client
	fallback   Embedder
	endpoint   string
	baseURL    string

	mu         sync.RWMutex
	lastStatus string
}

// NewClient creates an HTTP client for the embedding service.
// If fallback is nil, a default MockEmbedder is used when the remote service is unavailable.
func NewClient(cfg Config, fallback Embedder) *Client {
	if cfg.Dimension <= 0 {
		cfg.Dimension = model.DefaultVectorDim
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 500 * time.Millisecond
	}
	if fallback == nil {
		fallback = NewMockEmbedder(cfg.Dimension)
	}

	endpoint := cfg.URL
	baseURL := ""
	if endpoint != "" {
		if !strings.HasPrefix(endpoint, "http://") && !strings.HasPrefix(endpoint, "https://") {
			endpoint = "http://" + endpoint
		}
		baseURL = strings.TrimRight(endpoint, "/")
		if strings.HasSuffix(baseURL, "/v1/embeddings") {
			baseURL = strings.TrimSuffix(baseURL, "/v1/embeddings")
		} else {
			endpoint = baseURL + "/v1/embeddings"
		}
	}

	return &Client{
		cfg: cfg,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
		fallback:   fallback,
		endpoint:   endpoint,
		baseURL:    baseURL,
		lastStatus: "unknown",
	}
}

// NewClientFromEnv creates a Client initialised from environment variables.
func NewClientFromEnv() *Client {
	return NewClient(ConfigFromEnv(), nil)
}

// Dimension returns the vector dimension.
func (c *Client) Dimension() int {
	return c.cfg.Dimension
}

// URL returns the base URL of the service for telemetry reporting.
func (c *Client) URL() string {
	return c.baseURL
}

// IsEnabled returns whether embedding generation is enabled.
func (c *Client) IsEnabled() bool {
	return c.cfg.Enabled
}

// Fallback returns the fallback embedder.
func (c *Client) Fallback() Embedder {
	return c.fallback
}

// APIKey returns the configured API key (if any).
func (c *Client) APIKey() string {
	return c.cfg.APIKey
}

// EmbedText generates a dense vector embedding for a single string.
func (c *Client) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if !c.cfg.Enabled || c.endpoint == "" {
		if c.fallback != nil {
			return c.fallback.EmbedText(ctx, text)
		}
		return nil, errors.New("embedding engine is disabled and no fallback is configured")
	}

	res, err := c.embedHTTP(ctx, text)
	if err != nil {
		c.mu.Lock()
		c.lastStatus = "degraded"
		c.mu.Unlock()

		if c.fallback != nil {
			return c.fallback.EmbedText(ctx, text)
		}
		return nil, err
	}

	c.mu.Lock()
	c.lastStatus = "reachable"
	c.mu.Unlock()

	if len(res) == 0 {
		if c.fallback != nil {
			return c.fallback.EmbedText(ctx, text)
		}
		return nil, errors.New("empty embedding returned by service")
	}

	return res[0], nil
}

// EmbedBatch generates dense vector embeddings for multiple texts in a single batch request.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	if !c.cfg.Enabled || c.endpoint == "" {
		if c.fallback != nil {
			return c.fallback.EmbedBatch(ctx, texts)
		}
		return nil, errors.New("embedding engine is disabled and no fallback is configured")
	}

	res, err := c.embedHTTP(ctx, texts)
	if err != nil {
		c.mu.Lock()
		c.lastStatus = "degraded"
		c.mu.Unlock()

		if c.fallback != nil {
			return c.fallback.EmbedBatch(ctx, texts)
		}
		return nil, err
	}

	c.mu.Lock()
	c.lastStatus = "reachable"
	c.mu.Unlock()

	return res, nil
}

func (c *Client) embedHTTP(ctx context.Context, input any) ([][]float32, error) {
	reqBody, err := json.Marshal(openAIEmbeddingRequest{Input: input})
	if err != nil {
		return nil, fmt.Errorf("failed marshaling embedding request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed creating embedding http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("embedding service request failed: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed reading embedding response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding service returned HTTP %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var embResp openAIEmbeddingResponse
	if err := json.Unmarshal(bodyBytes, &embResp); err != nil {
		return nil, fmt.Errorf("failed parsing embedding JSON response: %w", err)
	}

	if embResp.Error != nil && embResp.Error.Message != "" {
		return nil, fmt.Errorf("embedding service error: %s", embResp.Error.Message)
	}

	if len(embResp.Data) == 0 {
		return nil, errors.New("no embedding data in service response")
	}

	// Sort by index to maintain exact input text ordering
	sort.Slice(embResp.Data, func(i, j int) bool {
		return embResp.Data[i].Index < embResp.Data[j].Index
	})

	result := make([][]float32, len(embResp.Data))
	for i, item := range embResp.Data {
		result[i] = item.Embedding
	}

	return result, nil
}

// Ping checks if the remote embedding service is reachable.
func (c *Client) Ping(ctx context.Context) error {
	if !c.cfg.Enabled || c.endpoint == "" {
		return errors.New("embedding engine is disabled or has no endpoint configured")
	}
	// A small probe text to verify endpoint responsiveness
	_, err := c.embedHTTP(ctx, "ping")
	return err
}

// CheckHealth probes the embedding service and returns current telemetry status.
func (c *Client) CheckHealth(ctx context.Context) model.EmbeddingEngineHealth {
	health := model.EmbeddingEngineHealth{
		Enabled:   c.cfg.Enabled,
		URL:       c.baseURL,
		Dimension: c.cfg.Dimension,
	}

	if !c.cfg.Enabled || c.endpoint == "" {
		health.Status = "offline"
		return health
	}

	// Quick probe to check reachability
	probeCtx, cancel := context.WithTimeout(ctx, 150*time.Millisecond)
	defer cancel()

	err := c.Ping(probeCtx)
	if err == nil {
		health.Status = "reachable"
		c.mu.Lock()
		c.lastStatus = "reachable"
		c.mu.Unlock()
	} else {
		if c.fallback != nil {
			health.Status = "degraded"
		} else {
			health.Status = "offline"
		}
		c.mu.Lock()
		c.lastStatus = health.Status
		c.mu.Unlock()
	}

	return health
}
