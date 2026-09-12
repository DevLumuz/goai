package openaicompat

import (
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"strings"
)

// Keep unknown fields as well as signatures and opaque encrypted blocks. These
// are protocol state, not merely display text. Never normalize their contents.
type reasoningDetailsAccumulator struct {
	details []map[string]any
	size    int64
}

var maxReasoningDetailsBytes int64 = 64 << 20

func (a *reasoningDetailsAccumulator) add(details []map[string]any) error {
	if details == nil {
		return nil
	}
	if a.details == nil {
		a.details = []map[string]any{}
	}
	for _, detail := range details {
		data, err := json.Marshal(detail)
		if err != nil {
			return fmt.Errorf("encoding reasoning detail: %w", err)
		}
		a.size += int64(len(data))
		if a.size > maxReasoningDetailsBytes {
			return fmt.Errorf("reasoning details exceed %d bytes", maxReasoningDetailsBytes)
		}
		var last map[string]any
		if len(a.details) > 0 {
			last = a.details[len(a.details)-1]
		}
		typ, _ := detail["type"].(string)
		// Consecutive text/summary fragments can share an index. An index can
		// also be reused across different types, so it is not a global identity.
		// Encrypted entries are discrete opaque blobs, never text fragments.
		if (typ == "reasoning.text" || typ == "reasoning.summary") && last != nil && last["type"] == typ && sameReasoningDetail(last, detail) {
			key := "text"
			if typ == "reasoning.summary" {
				key = "summary"
			}
			for k, v := range detail {
				if k == key {
					old, _ := last[k].(string)
					next, _ := v.(string)
					last[k] = old + next
				} else if v != nil || last[k] == nil {
					last[k] = v
				}
			}
		} else {
			a.details = append(a.details, maps.Clone(detail))
		}
	}
	return nil
}

func sameReasoningDetail(a, b map[string]any) bool {
	for _, key := range []string{"id", "index", "format"} {
		if a[key] != nil && b[key] != nil && !reflect.DeepEqual(a[key], b[key]) {
			return false
		}
	}
	return true
}

func reasoningDetailsText(details []map[string]any) string {
	var text strings.Builder
	for _, detail := range details {
		switch detail["type"] {
		case "reasoning.text":
			s, _ := detail["text"].(string)
			text.WriteString(s)
		case "reasoning.summary":
			s, _ := detail["summary"].(string)
			text.WriteString(s)
		}
	}
	return text.String()
}

func reasoningDetailsArray(value any) []any {
	switch v := value.(type) {
	case []any:
		return v
	case []map[string]any:
		if v == nil {
			return nil
		}
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = item
		}
		return out
	default:
		return nil
	}
}
