package google

import (
	"fmt"
	"strings"

	"github.com/zendev-sh/goai/provider"
)

// Live fields that only one backend accepts, as the official SDK's converters
// enforce them. Paths are relative to the setup message; a path crosses arrays
// element by element. Each list is checked before anything is sent, so a field
// the backend would reject or silently ignore fails with ErrRealtimeUnsupported.
var (
	liveVertexOnlyFields = []string{
		"sessionResumption.transparent",
		"explicitVadSignal",
		"safetySettings.method",
		"tools.retrieval",
		"tools.enterpriseWebSearch",
		"tools.parallelAiSearch",
		"tools.exaAiSearch",
		"tools.googleSearch.excludeDomains",
		"tools.googleSearch.blockingConfidence",
	}
	liveGeminiAPIOnlyFields = []string{
		"generationConfig.speechConfig.voiceConfig.replicatedVoiceConfig.consentAudio",
		"generationConfig.speechConfig.voiceConfig.replicatedVoiceConfig.voiceConsentSignature",
		"tools.fileSearch",
		"tools.mcpServers",
	}
	liveUnsupportedFields = []string{
		"generationConfig.speechConfig.multiSpeakerVoiceConfig",
	}
)

// checkLiveBackend rejects the fields of msg that the backend does not accept.
// The error names the constructor of the backend that does.
func checkLiveBackend(msg map[string]any, isVertex bool) error {
	for _, path := range liveUnsupportedFields {
		if livePathSet(msg, strings.Split(path, ".")) {
			return fmt.Errorf("%w: %s is not supported by the Live API", provider.ErrRealtimeUnsupported, path)
		}
	}
	rejected, backend, constructor := liveVertexOnlyFields, "Vertex AI", "vertex.Live"
	if isVertex {
		rejected, backend, constructor = liveGeminiAPIOnlyFields, "the Gemini API", "google.Live"
	}
	for _, path := range rejected {
		if livePathSet(msg, strings.Split(path, ".")) {
			return fmt.Errorf("%w: %s is only supported on %s (%s)", provider.ErrRealtimeUnsupported, path, backend, constructor)
		}
	}
	return nil
}

// livePathSet reports whether path holds a value in v, in any element of the
// arrays it crosses.
func livePathSet(v any, path []string) bool {
	if len(path) == 0 {
		return v != nil
	}
	switch node := v.(type) {
	case map[string]any:
		return livePathSet(node[path[0]], path[1:])
	case []map[string]any:
		for _, item := range node {
			if livePathSet(item, path) {
				return true
			}
		}
	case []any:
		for _, item := range node {
			if livePathSet(item, path) {
				return true
			}
		}
	}
	return false
}
