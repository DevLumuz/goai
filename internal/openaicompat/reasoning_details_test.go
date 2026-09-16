package openaicompat

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

func TestReasoningDetailsBoundaries(t *testing.T) {
	var a reasoningDetailsAccumulator
	input := []map[string]any{
		{"type": "reasoning.text", "text": "one", "id": "a", "index": 0},
		{"type": "reasoning.text", "text": "two", "id": "b", "index": 1},
		{"type": "reasoning.text", "signature": "sig", "id": "b", "index": 1},
		{"type": "reasoning.summary", "summary": "sum", "index": 1},
		{"type": "reasoning.encrypted", "data": "x", "index": 1},
		{"type": "reasoning.encrypted", "data": "y", "index": 1},
		{"type": "reasoning.future", "payload": map[string]any{"nested": true}},
	}
	before, _ := json.Marshal(input)
	if err := a.add(input); err != nil {
		t.Fatal(err)
	}
	if len(a.details) != 6 || a.details[1]["text"] != "two" || a.details[1]["signature"] != "sig" {
		t.Fatalf("wrong assembly: %#v", a.details)
	}
	after, _ := json.Marshal(input)
	if string(before) != string(after) {
		t.Fatal("mutated input")
	}
	if err := a.add([]map[string]any{{"type": "reasoning.text", "id": map[string]any{"bad": true}}, {"type": "reasoning.text", "id": map[string]any{"different": true}}}); err != nil {
		t.Fatal(err)
	}
}

func TestReasoningDetailsFragmentAssembly(t *testing.T) {
	var a reasoningDetailsAccumulator
	fragments := []map[string]any{
		{"type": "reasoning.text", "text": "thi", "signature": "sig", "id": "r", "index": 0},
		{"type": "reasoning.text", "text": "nk", "signature": nil, "id": "r", "index": 0},
		{"type": "reasoning.summary", "summary": "sum", "index": 0},
		{"type": "reasoning.summary", "summary": "mary", "index": 0},
		{"type": "reasoning.encrypted", "data": "one", "index": 0},
		{"type": "reasoning.encrypted", "data": "two", "index": 0},
	}
	before, _ := json.Marshal(fragments)
	if err := a.add(fragments); err != nil {
		t.Fatal(err)
	}
	want := []map[string]any{
		{"type": "reasoning.text", "text": "think", "signature": "sig", "id": "r", "index": 0},
		{"type": "reasoning.summary", "summary": "summary", "index": 0},
		{"type": "reasoning.encrypted", "data": "one", "index": 0},
		{"type": "reasoning.encrypted", "data": "two", "index": 0},
	}
	if !reflect.DeepEqual(a.details, want) {
		t.Fatalf("details=%#v want %#v", a.details, want)
	}
	after, _ := json.Marshal(fragments)
	if string(before) != string(after) {
		t.Fatal("assembly mutated fragments")
	}
}

func TestReasoningDetailsRejectsNonJSONState(t *testing.T) {
	var a reasoningDetailsAccumulator
	err := a.add([]map[string]any{{"type": "reasoning.future", "payload": func() {}}})
	var unsupported *json.UnsupportedTypeError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error=%v, want wrapped unsupported JSON type", err)
	}
	if len(a.details) != 0 {
		t.Fatal("invalid state appended")
	}
}

func TestReasoningDetailsPartRepresentations(t *testing.T) {
	detail := map[string]any{"type": "reasoning.encrypted", "data": "opaque"}
	for _, tc := range []struct {
		name       string
		value      any
		structured bool
	}{
		{"typed", []map[string]any{detail}, true},
		{"persisted", []any{detail}, true},
		{"empty", []any{}, true},
		{"typed-nil", []map[string]any(nil), false},
		{"invalid-type", "not an array", false},
		{"missing", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parts := []provider.Part{{Type: provider.PartReasoning, Text: "think", ProviderOptions: map[string]any{"openrouter": map[string]any{"reasoning_details": tc.value}}}}
			body := ConvertMessagesWithConfig([]provider.Message{{Role: provider.RoleAssistant, Content: parts}}, "", MessagesConfig{IncludeReasoningContent: true, IncludeReasoningDetails: true})[0]
			if tc.structured {
				want := []any{detail}
				if tc.name == "empty" {
					want = []any{}
				}
				if !reflect.DeepEqual(body["reasoning_details"], want) {
					t.Fatalf("details=%#v want %#v", body["reasoning_details"], want)
				}
				if body["reasoning"] != nil {
					t.Fatal("structured replay also emitted plaintext")
				}
			} else if body["reasoning"] != "think" {
				t.Fatalf("plaintext fallback=%#v", body)
			}
			if body["reasoning_content"] != nil {
				t.Fatal("both plaintext aliases sent")
			}
		})
	}
}

func TestReasoningDetailsReplayOptIn(t *testing.T) {
	details := []map[string]any{{"type": "reasoning.text", "text": "think", "signature": "sig"}}
	messages := []provider.Message{{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartReasoning, Text: "think", ProviderOptions: map[string]any{"openrouter": map[string]any{"reasoning_details": details}}}}}}
	for _, cfg := range []MessagesConfig{{}, {IncludeReasoningContent: true}} {
		body := ConvertMessagesWithConfig(messages, "", cfg)[0]
		if _, ok := body["reasoning_details"]; ok {
			t.Fatal("details leaked to non-opt-in provider")
		}
		if cfg.IncludeReasoningContent && body["reasoning_content"] != "think" {
			t.Fatal("existing reasoning_content behavior changed")
		}
	}
	// Message metadata takes precedence over part metadata and is not duplicated.
	messages[0].ProviderOptions = map[string]any{"openrouter": map[string]any{"reasoning_details": details}}
	body := ConvertMessagesWithConfig(messages, "", MessagesConfig{IncludeReasoningDetails: true})[0]
	if !reflect.DeepEqual(body["reasoning_details"], []any{details[0]}) {
		t.Fatalf("duplicated details: %#v", body)
	}
}

func TestReasoningDetailsStreamLimit(t *testing.T) {
	old := maxReasoningDetailsBytes
	maxReasoningDetailsBytes = 10
	t.Cleanup(func() { maxReasoningDetailsBytes = old })
	out := make(chan provider.StreamChunk, 5)
	go ParseStream(t.Context(), sse.NewScanner(strings.NewReader("data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.encrypted\",\"data\":\"long\"}]}}]}\n\n")), out)
	var gotError bool
	for chunk := range out {
		if chunk.Type == provider.ChunkError {
			gotError = true
		}
		if chunk.Type == provider.ChunkFinish {
			t.Fatal("oversized state silently accepted")
		}
	}
	if !gotError {
		t.Fatal("missing size error")
	}
}
