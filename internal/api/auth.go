package api

import (
	"crypto/subtle"
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"sync"
)

var warnDevAuthOnce sync.Once

// AuthMiddleware creates an HTTP middleware that verifies API keys or Bearer tokens against apiKey.
// If apiKey is empty, it operates in permissive dev mode with a logged warning.
// Public paths (e.g. /health, /api/v1/memory/health, /api/v1/consolidation/health) are exempt from authentication.
func AuthMiddleware(apiKey string, publicPaths ...string) func(http.Handler) http.Handler {
	trimmedKey := strings.TrimSpace(apiKey)
	if trimmedKey == "" {
		warnDevAuthOnce.Do(func() {
			log.Println("[SECURITY WARNING] SEKHA_API_KEY is not configured; running in permissive / dev mode (unauthenticated)")
		})
	}

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// OPTIONS requests for CORS are allowed through
			if r.Method == http.MethodOptions {
				next.ServeHTTP(w, r)
				return
			}

			// If no API key configured, pass through in permissive dev mode
			if trimmedKey == "" {
				next.ServeHTTP(w, r)
				return
			}

			// Check public paths exemption
			reqPath := r.URL.Path
			for _, pub := range publicPaths {
				cleanPub := strings.TrimSpace(pub)
				if reqPath == cleanPub || strings.TrimSuffix(reqPath, "/") == strings.TrimSuffix(cleanPub, "/") {
					next.ServeHTTP(w, r)
					return
				}
			}

			// Extract API Key from X-API-Key or Authorization: Bearer <key>
			var token string
			if keyHeader := r.Header.Get("X-API-Key"); keyHeader != "" {
				token = strings.TrimSpace(keyHeader)
			} else if authHeader := r.Header.Get("Authorization"); authHeader != "" {
				if strings.HasPrefix(strings.ToLower(authHeader), "bearer ") {
					token = strings.TrimSpace(authHeader[7:])
				}
			}

			if token == "" || subtle.ConstantTimeCompare([]byte(token), []byte(trimmedKey)) != 1 {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusUnauthorized)
				_ = json.NewEncoder(w).Encode(map[string]string{
					"error": "unauthorized: invalid or missing API key",
				})
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}
