package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

const (
	// CipherPrefix identifies AES-256-GCM encrypted payloads in Sekha.
	CipherPrefix = "enc:v1:"

	// RedactedMask is the redaction replacement for masked secret values.
	RedactedMask = "[REDACTED_SECRET]"

	// DefaultMasterKeyFile defines the standard POSIX master key path on Node 1.
	DefaultMasterKeyFile = "/etc/sekha/master.key"

	// DevFallbackMasterKey is the deterministic master key used only in development/testing mode.
	DevFallbackMasterKey = "sekha-dev-fallback-insecure-master-key-seed-32bytes"
)

// Cipher defines authenticated encryption, decryption, and secret masking operations.
type Cipher interface {
	Encrypt(plaintext string) (string, error)
	Decrypt(ciphertext string) (string, error)
	IsEncrypted(val string) bool
	Mask(val string) string
}

// AESGCMCipher implements Cipher using AES-256-GCM with a SHA-256-derived 256-bit key.
type AESGCMCipher struct {
	key []byte // 32 bytes for AES-256
}

var _ Cipher = (*AESGCMCipher)(nil)

// IsVaultReference checks whether a string is a secret manager / vault URI reference
// (e.g. vault://... or env://...) that should be preserved without encryption.
func IsVaultReference(val string) bool {
	return strings.HasPrefix(val, "vault://") || strings.HasPrefix(val, "env://")
}

// resolveMasterKey resolves the raw master key from the argument, environment, or file.
func resolveMasterKey(masterKey string) string {
	if strings.TrimSpace(masterKey) != "" {
		return strings.TrimSpace(masterKey)
	}

	if envKey := strings.TrimSpace(os.Getenv("SEKHA_MASTER_KEY")); envKey != "" {
		return envKey
	}

	if info, err := os.Stat(DefaultMasterKeyFile); err == nil {
		if info.Mode().Perm()&0077 != 0 {
			log.Printf("[SECURITY WARNING] %s permissions are %#o, recommended 0600", DefaultMasterKeyFile, info.Mode().Perm())
		}
		data, err := os.ReadFile(DefaultMasterKeyFile)
		if err == nil {
			trimmed := strings.TrimSpace(string(data))
			if trimmed != "" {
				return trimmed
			}
		}
	}

	log.Println("[SECURITY WARNING] No master key configured via SEKHA_MASTER_KEY or /etc/sekha/master.key; using deterministic development fallback key. DO NOT USE IN PRODUCTION!")
	return DevFallbackMasterKey
}

// NewAESGCMCipher creates a new AES-256-GCM cipher deriving a 32-byte key from masterKey.
// If masterKey is empty, it checks SEKHA_MASTER_KEY, then /etc/sekha/master.key, or falls back to a dev key.
func NewAESGCMCipher(masterKey string) (*AESGCMCipher, error) {
	resolved := resolveMasterKey(masterKey)
	h := sha256.Sum256([]byte(resolved))
	return &AESGCMCipher{key: h[:]}, nil
}

// NewAESGCMCipherFromKey creates an AESGCMCipher directly with a 32-byte key.
func NewAESGCMCipherFromKey(key []byte) (*AESGCMCipher, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("AES-256 requires a 32-byte key; received %d bytes", len(key))
	}
	keyCopy := make([]byte, 32)
	copy(keyCopy, key)
	return &AESGCMCipher{key: keyCopy}, nil
}

// IsEncrypted returns true if the value starts with the enc:v1: ciphertext prefix.
func (c *AESGCMCipher) IsEncrypted(val string) bool {
	return strings.HasPrefix(val, CipherPrefix)
}

// Mask returns the standard secret redaction placeholder.
func (c *AESGCMCipher) Mask(val string) string {
	return RedactedMask
}

// Encrypt encrypts plaintext using AES-256-GCM with a 12-byte random nonce.
// If the string is already encrypted, or is a vault/env URI reference, it is returned unchanged.
func (c *AESGCMCipher) Encrypt(plaintext string) (string, error) {
	if plaintext == "" {
		return "", nil
	}
	if c.IsEncrypted(plaintext) || IsVaultReference(plaintext) {
		return plaintext, nil
	}

	block, err := aes.NewCipher(c.key)
	if err != nil {
		return "", fmt.Errorf("failed to initialise AES cipher block: %w", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", fmt.Errorf("failed to initialise GCM AEAD: %w", err)
	}

	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return "", fmt.Errorf("failed to generate random nonce: %w", err)
	}

	// Seal appends ciphertext and 16-byte tag to nonce
	sealed := gcm.Seal(nonce, nonce, []byte(plaintext), nil)
	return CipherPrefix + base64.StdEncoding.EncodeToString(sealed), nil
}

// Decrypt parses an enc:v1: ciphertext, extracts the nonce and tag, and decrypts the payload.
// If the input does not have the enc:v1: prefix, it is returned unchanged.
func (c *AESGCMCipher) Decrypt(ciphertext string) (string, error) {
	if ciphertext == "" {
		return "", nil
	}
	if !c.IsEncrypted(ciphertext) {
		return ciphertext, nil
	}

	b64Data := strings.TrimPrefix(ciphertext, CipherPrefix)
	data, err := base64.StdEncoding.DecodeString(b64Data)
	if err != nil {
		return "", fmt.Errorf("invalid base64 encoding in ciphertext: %w", err)
	}

	block, err := aes.NewCipher(c.key)
	if err != nil {
		return "", fmt.Errorf("failed to initialise AES cipher block: %w", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", fmt.Errorf("failed to initialise GCM AEAD: %w", err)
	}

	nonceSize := gcm.NonceSize()
	overhead := gcm.Overhead()
	if len(data) < nonceSize+overhead {
		return "", fmt.Errorf("ciphertext payload too short: got %d bytes, need at least %d", len(data), nonceSize+overhead)
	}

	nonce := data[:nonceSize]
	payload := data[nonceSize:]

	plaintextBytes, err := gcm.Open(nil, nonce, payload, nil)
	if err != nil {
		return "", fmt.Errorf("decryption failed / authentication tag mismatch: %w", err)
	}

	return string(plaintextBytes), nil
}
