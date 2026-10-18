package realtimews

import (
	"bytes"
	"encoding/json"
)

// IsJSONObject reports whether data is one valid JSON object, the shape of
// every message the realtime APIs accept.
func IsJSONObject(data []byte) bool {
	trimmed := bytes.TrimSpace(data)
	return len(trimmed) > 0 && trimmed[0] == '{' && json.Valid(trimmed)
}
