package security

import (
	"encoding/base64"
	"strings"
	"testing"
)

func TestAESGCMCipher_RoundTrip(t *testing.T) {
	cipher, err := NewAESGCMCipher("test-master-key-12345")
	if err != nil {
		t.Fatalf("failed to create cipher: %v", err)
	}

	testCases := []struct {
		name      string
		plaintext string
	}{
		{"API Key", "API_KEY=hvb_live_999 secret_token"},
		{"Short string", "my-secret"},
		{"Unicode and emojis", "🔐 Top Secret 🔑 秘密 123!"},
		{"JSON payload", `{"client_id": "sekha-node1", "token": "xyz789"}`},
		{"Multiline string", "LINE 1: key\nLINE 2: cert\nLINE 3: token"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			encrypted, err := cipher.Encrypt(tc.plaintext)
			if err != nil {
				t.Fatalf("encryption failed: %v", err)
			}

			if !strings.HasPrefix(encrypted, "enc:v1:") {
				t.Errorf("expected ciphertext to start with 'enc:v1:', got: %s", encrypted)
			}

			if !cipher.IsEncrypted(encrypted) {
				t.Errorf("IsEncrypted returned false for %s", encrypted)
			}

			if strings.Contains(encrypted, tc.plaintext) {
				t.Errorf("ciphertext leaked plaintext %q in %s", tc.plaintext, encrypted)
			}

			decrypted, err := cipher.Decrypt(encrypted)
			if err != nil {
				t.Fatalf("decryption failed: %v", err)
			}

			if decrypted != tc.plaintext {
				t.Errorf("decrypted text mismatch:\nwant: %s\ngot:  %s", tc.plaintext, decrypted)
			}
		})
	}
}

func TestAESGCMCipher_EmptyString(t *testing.T) {
	cipher, err := NewAESGCMCipher("test-master-key")
	if err != nil {
		t.Fatalf("failed to create cipher: %v", err)
	}

	enc, err := cipher.Encrypt("")
	if err != nil || enc != "" {
		t.Errorf("expected empty string encrypt to return empty, got %q, err=%v", enc, err)
	}

	dec, err := cipher.Decrypt("")
	if err != nil || dec != "" {
		t.Errorf("expected empty string decrypt to return empty, got %q, err=%v", dec, err)
	}
}

func TestAESGCMCipher_NonceRandomness(t *testing.T) {
	cipher, err := NewAESGCMCipher("test-master-key")
	if err != nil {
		t.Fatalf("failed to create cipher: %v", err)
	}

	plaintext := "sensitive_credential_repeat_check"
	c1, err := cipher.Encrypt(plaintext)
	if err != nil {
		t.Fatal(err)
	}
	c2, err := cipher.Encrypt(plaintext)
	if err != nil {
		t.Fatal(err)
	}

	if c1 == c2 {
		t.Errorf("AES-GCM produced identical ciphertexts for identical plaintext; nonces must be unique! c1=%s, c2=%s", c1, c2)
	}
}

func TestAESGCMCipher_VaultReferencesPreserved(t *testing.T) {
	cipher, err := NewAESGCMCipher("test-master-key")
	if err != nil {
		t.Fatalf("failed to create cipher: %v", err)
	}

	vaultRefs := []string{
		"vault://secrets/kestrel-api-key",
		"vault://secrets/node1/tls-cert",
		"env://SEKHA_CLUSTER_TOKEN",
		"env://DATABASE_URL",
	}

	for _, ref := range vaultRefs {
		if !IsVaultReference(ref) {
			t.Errorf("expected IsVaultReference to be true for %s", ref)
		}

		enc, err := cipher.Encrypt(ref)
		if err != nil {
			t.Fatalf("Encrypt failed on vault ref %s: %v", ref, err)
		}
		if enc != ref {
			t.Errorf("Encrypt altered vault reference:\nwant: %s\ngot:  %s", ref, enc)
		}

		dec, err := cipher.Decrypt(ref)
		if err != nil {
			t.Fatalf("Decrypt failed on vault ref %s: %v", ref, err)
		}
		if dec != ref {
			t.Errorf("Decrypt altered vault reference:\nwant: %s\ngot:  %s", ref, dec)
		}
	}
}

func TestAESGCMCipher_NoDoubleEncryption(t *testing.T) {
	cipher, err := NewAESGCMCipher("test-master-key")
	if err != nil {
		t.Fatalf("failed to create cipher: %v", err)
	}

	plaintext := "secret-double-encrypt-test"
	enc1, err := cipher.Encrypt(plaintext)
	if err != nil {
		t.Fatal(err)
	}

	// Encrypt again
	enc2, err := cipher.Encrypt(enc1)
	if err != nil {
		t.Fatal(err)
	}

	if enc1 != enc2 {
		t.Errorf("expected double-encryption to be a no-op; got %s != %s", enc1, enc2)
	}
}

func TestAESGCMCipher_DecryptUnencrypted(t *testing.T) {
	cipher, err := NewAESGCMCipher("test-master-key")
	if err != nil {
		t.Fatalf("failed to create cipher: %v", err)
	}

	plain := "plain-unencrypted-summary"
	dec, err := cipher.Decrypt(plain)
	if err != nil {
		t.Fatalf("unexpected error decrypting unencrypted string: %v", err)
	}
	if dec != plain {
		t.Errorf("expected %s, got %s", plain, dec)
	}
}

func TestAESGCMCipher_TamperDetection(t *testing.T) {
	cipher, err := NewAESGCMCipher("test-master-key")
	if err != nil {
		t.Fatalf("failed to create cipher: %v", err)
	}

	plaintext := "tamper-check-payload-99"
	enc, err := cipher.Encrypt(plaintext)
	if err != nil {
		t.Fatal(err)
	}

	rawB64 := strings.TrimPrefix(enc, CipherPrefix)
	rawBytes, err := base64.StdEncoding.DecodeString(rawB64)
	if err != nil {
		t.Fatal(err)
	}

	// Flip a byte in the payload / tag
	tamperedBytes := make([]byte, len(rawBytes))
	copy(tamperedBytes, rawBytes)
	tamperedBytes[len(tamperedBytes)-1] ^= 0x01

	tamperedCiphertext := CipherPrefix + base64.StdEncoding.EncodeToString(tamperedBytes)

	_, err = cipher.Decrypt(tamperedCiphertext)
	if err == nil {
		t.Errorf("expected decryption to fail for tampered ciphertext, but got nil error")
	}

	// Invalid base64
	_, err = cipher.Decrypt(CipherPrefix + "not-valid-base-64!!")
	if err == nil {
		t.Errorf("expected decryption to fail for invalid base64, but got nil error")
	}

	// Short payload
	shortBytes := []byte("short")
	shortCipher := CipherPrefix + base64.StdEncoding.EncodeToString(shortBytes)
	_, err = cipher.Decrypt(shortCipher)
	if err == nil {
		t.Errorf("expected decryption to fail for short payload, but got nil error")
	}
}

func TestAESGCMCipher_WrongKeyFails(t *testing.T) {
	cipher1, _ := NewAESGCMCipher("master-key-node1")
	cipher2, _ := NewAESGCMCipher("master-key-node2")

	plaintext := "classified-cluster-credentials"
	enc, err := cipher1.Encrypt(plaintext)
	if err != nil {
		t.Fatal(err)
	}

	_, err = cipher2.Decrypt(enc)
	if err == nil {
		t.Errorf("expected decryption to fail with wrong master key, but it succeeded")
	}
}

func TestAESGCMCipher_Mask(t *testing.T) {
	cipher, _ := NewAESGCMCipher("test-key")
	masked := cipher.Mask("super-secret-summary")
	if masked != "[REDACTED_SECRET]" {
		t.Errorf("expected '[REDACTED_SECRET]', got %q", masked)
	}
}

func TestAESGCMCipher_DevFallbackKey(t *testing.T) {
	// Call NewAESGCMCipher with empty string
	cipher, err := NewAESGCMCipher("")
	if err != nil {
		t.Fatalf("failed with empty master key: %v", err)
	}

	plaintext := "fallback-test"
	enc, err := cipher.Encrypt(plaintext)
	if err != nil {
		t.Fatal(err)
	}

	dec, err := cipher.Decrypt(enc)
	if err != nil {
		t.Fatal(err)
	}

	if dec != plaintext {
		t.Errorf("expected %s, got %s", plaintext, dec)
	}
}
