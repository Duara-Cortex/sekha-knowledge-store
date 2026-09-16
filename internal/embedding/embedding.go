package embedding

import (
	"hash/fnv"
	"math"
	"regexp"
	"strings"
)

// Common English and technical stop words filtered during token extraction.
var stopWords = map[string]bool{
	"a": true, "about": true, "above": true, "after": true, "again": true, "all": true,
	"am": true, "an": true, "and": true, "any": true, "are": true, "aren't": true,
	"as": true, "at": true, "be": true, "because": true, "been": true, "before": true,
	"being": true, "below": true, "between": true, "both": true, "but": true, "by": true,
	"can": true, "cannot": true, "could": true, "did": true, "do": true, "does": true,
	"doing": true, "down": true, "during": true, "each": true, "few": true, "for": true,
	"from": true, "further": true, "had": true, "has": true, "have": true, "having": true,
	"he": true, "her": true, "here": true, "hers": true, "herself": true, "him": true,
	"himself": true, "his": true, "how": true, "i": true, "if": true, "in": true,
	"into": true, "is": true, "isn't": true, "it": true, "its": true, "itself": true,
	"let's": true, "me": true, "more": true, "most": true, "my": true, "myself": true,
	"no": true, "nor": true, "not": true, "of": true, "off": true, "on": true,
	"once": true, "only": true, "or": true, "other": true, "ought": true, "our": true,
	"ours": true, "ourselves": true, "out": true, "over": true, "own": true, "same": true,
	"she": true, "should": true, "so": true, "some": true, "such": true, "than": true,
	"that": true, "the": true, "their": true, "theirs": true, "them": true, "themselves": true,
	"then": true, "there": true, "these": true, "they": true, "this": true, "those": true,
	"through": true, "to": true, "too": true, "under": true, "until": true, "up": true,
	"very": true, "was": true, "wasn't": true, "we": true, "were": true, "weren't": true,
	"what": true, "when": true, "where": true, "which": true, "while": true, "who": true,
	"whom": true, "why": true, "with": true, "won't": true, "would": true, "you": true,
	"your": true, "yours": true, "yourself": true, "yourselves": true,
}

var tokenRegexp = regexp.MustCompile(`[a-zA-Z0-9_\-\.:]{2,}`)

// ExtractTokens tokenizes text into lowercased distinctive tokens without stop words.
func ExtractTokens(text string) []string {
	matches := tokenRegexp.FindAllString(text, -1)
	var tokens []string
	seen := make(map[string]bool)

	for _, m := range matches {
		clean := strings.ToLower(strings.Trim(m, ".,:;()[]\"'_-"))
		if len(clean) < 2 {
			continue
		}
		if stopWords[clean] {
			continue
		}
		if !seen[clean] {
			seen[clean] = true
			tokens = append(tokens, clean)
		}
	}
	return tokens
}

// projectToken maps a single token string deterministically into a D-dimensional direction.
// Uses FNV-1a 64-bit hash and a 64-bit pseudo-random generator (xorshift64star).
// In accordance with Random Indexing / Johnson-Lindenstrauss lemma, distinct tokens
// receive pseudo-orthogonal random directions, while identical tokens receive identical vectors.
func projectToken(token string, dims int, weight float32, vec []float32) {
	h := fnv.New64a()
	_, _ = h.Write([]byte(token))
	state := h.Sum64()
	if state == 0 {
		state = 0x517cc1b727220a95
	}

	for i := 0; i < dims; i++ {
		// xorshift64star
		state ^= state >> 12
		state ^= state << 25
		state ^= state >> 27
		val := state * 0x2545F4914F6CDD1D

		// Map to float in [-1.0, 1.0]
		f := float32(int32(val>>32)) / 2147483648.0
		vec[i] += f * weight
	}
}

// extractStem performs lightweight suffix normalization for English inflections.
func extractStem(word string) string {
	if len(word) <= 3 {
		return word
	}
	stem := word
	// Common plural / inflection suffixes
	suffixes := []string{"ations", "ation", "tions", "tion", "ings", "ing", "ments", "ment", "ness", "sses", "ies", "ed", "es", "s"}
	for _, suf := range suffixes {
		if strings.HasSuffix(stem, suf) && len(stem)-len(suf) >= 3 {
			stem = stem[:len(stem)-len(suf)]
			if suf == "ies" {
				stem += "y"
			}
			break
		}
	}
	// Normalise trailing 'e' (e.g. consolidate -> consolidat, trace -> trac)
	if len(stem) > 3 && strings.HasSuffix(stem, "e") {
		stem = strings.TrimSuffix(stem, "e")
	}
	return stem
}

// Generate synthesises a deterministic, unit-normalised float32 vector
// from input text using token, stem, and subword n-gram feature projections.
// Unlike whole-string SHA-256 projections, texts with overlapping words and
// concepts yield high cosine similarity, while unrelated texts yield near-zero similarity.
func Generate(text string, dims int) []float32 {
	if dims <= 0 {
		dims = 64
	}

	vec := make([]float32, dims)
	tokens := ExtractTokens(text)
	if len(tokens) == 0 {
		// Fallback for empty or purely stopword input
		raw := strings.ToLower(strings.TrimSpace(text))
		if raw != "" {
			projectToken(raw, dims, 1.0, vec)
		} else {
			return vec
		}
	}

	for i, token := range tokens {
		// 1. Morphological Stem Projection (primary semantic root anchor)
		stem := extractStem(token)
		wordWeight := float32(1.0 + 0.15*math.Min(5.0, float64(len(token))))
		projectToken("s:"+stem, dims, wordWeight*1.2, vec)

		// 2. Primary Word Token Projection (distinguishes exact surface form)
		projectToken("w:"+token, dims, wordWeight*0.7, vec)

		// 3. Subword character 3-grams and 4-grams (captures subword similarities)
		runes := []rune(token)
		if len(runes) >= 3 {
			for j := 0; j <= len(runes)-3; j++ {
				ngram := string(runes[j : j+3])
				projectToken("ng3:"+ngram, dims, 0.40, vec)
			}
		}
		if len(runes) >= 4 {
			for j := 0; j <= len(runes)-4; j++ {
				ngram := string(runes[j : j+4])
				projectToken("ng4:"+ngram, dims, 0.35, vec)
			}
		}

		// 4. Adjacent word bigrams (captures local phrase semantics)
		if i+1 < len(tokens) {
			bigram := token + "_" + tokens[i+1]
			projectToken("bi:"+bigram, dims, 0.5, vec)
		}
	}

	// Normalise to unit Euclidean length
	var sum float64
	for _, v := range vec {
		sum += float64(v * v)
	}
	mag := float32(math.Sqrt(sum))
	if mag > 0 {
		for i := range vec {
			vec[i] /= mag
		}
	}

	return vec
}

// CosineSimilarity calculates the dot product between two normalised float32 vectors,
// clamping the result to [0.0, 1.0].
func CosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0.0
	}
	var dot float64
	for i := range a {
		dot += float64(a[i] * b[i])
	}
	if dot > 1.0 {
		return 1.0
	}
	if dot < 0.0 {
		return 0.0
	}
	return dot
}

// ScoreLexical computes a lexical and keyword matching score in [0.0, 1.0]
// between query tokens and the target entity's label and summary.
func ScoreLexical(queryTokens []string, label, summary string) float64 {
	if len(queryTokens) == 0 {
		return 0.0
	}

	lowerLabel := strings.ToLower(label)
	lowerSummary := strings.ToLower(summary)

	labelTokens := ExtractTokens(lowerLabel)
	summaryTokens := ExtractTokens(lowerSummary)

	labelSet := make(map[string]bool, len(labelTokens))
	for _, t := range labelTokens {
		labelSet[t] = true
	}

	summarySet := make(map[string]bool, len(summaryTokens))
	for _, t := range summaryTokens {
		summarySet[t] = true
	}

	matchedWeight := 0.0
	totalWeight := float64(len(queryTokens)) * 2.0 // Label match weight baseline

	for _, qt := range queryTokens {
		if labelSet[qt] {
			matchedWeight += 2.0 // Exact word in label
		} else if strings.Contains(lowerLabel, qt) {
			matchedWeight += 1.2 // Substring match in label
		} else if summarySet[qt] {
			matchedWeight += 1.0 // Exact word in summary
		} else if strings.Contains(lowerSummary, qt) {
			matchedWeight += 0.5 // Substring match in summary
		}
	}

	ratio := matchedWeight / totalWeight
	if ratio > 1.0 {
		ratio = 1.0
	}

	// Full query string exact match bonus
	queryJoined := strings.Join(queryTokens, " ")
	if strings.Contains(lowerLabel, queryJoined) {
		ratio = math.Max(ratio, 0.95)
	} else if strings.Contains(lowerSummary, queryJoined) {
		ratio = math.Max(ratio, 0.80)
	}

	return math.Round(ratio*10000) / 10000
}
