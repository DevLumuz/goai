package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

func TestThinkingBindingRequest(t *testing.T) {
	for _, tc := range []struct {
		name     string
		thinking map[string]any
		behavior string
	}{
		{"adaptive", map[string]any{"type": "adaptive", "blockBinding": map[string]any{"prefixMismatchBehavior": "drop_block"}}, "drop_block"},
		{"binding-only", map[string]any{"blockBinding": map[string]any{"prefixMismatchBehavior": "error"}}, "error"},
		{"wire", map[string]any{"type": "enabled", "budgetTokens": 1024, "block_binding": map[string]any{"prefix_mismatch_behavior": "drop_block"}}, "drop_block"},
		{"disabled", map[string]any{"type": "disabled"}, ""},
		{"unspecified", nil, ""},
	} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%v", tc.name, streaming), func(t *testing.T) {
				before, _ := json.Marshal(tc.thinking)
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
						return
					}
					thinking, _ := body["thinking"].(map[string]any)
					binding, _ := thinking["block_binding"].(map[string]any)
					if tc.behavior != "" && binding["prefix_mismatch_behavior"] != tc.behavior {
						t.Errorf("thinking=%#v", thinking)
					}
					if tc.behavior == "" && binding != nil {
						t.Errorf("unsolicited binding: %#v", binding)
					}
					if _, ok := thinking["blockBinding"]; ok {
						t.Error("camelCase key leaked")
					}
					beta := r.Header.Get("anthropic-beta")
					if strings.Contains(beta, "thinking-binding-controls-2026-08-01") != (tc.behavior != "") {
						t.Errorf("beta=%q", beta)
					}
					if !strings.Contains(beta, "caller-beta") {
						t.Error("caller beta lost")
					}
					if body["stream"] == true {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"content\":[],\"usage\":{}}}\n\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\ndata: {\"type\":\"message_stop\"}\n\n")
					} else {
						fmt.Fprint(w, `{"id":"m","content":[],"stop_reason":"end_turn","usage":{}}`)
					}
				}))
				defer srv.Close()
				model := Chat("claude-fable-5-1", WithAPIKey("test"), WithBaseURL(srv.URL), WithAutoStreaming(false), WithHeaders(map[string]string{"anthropic-beta": "caller-beta"}))
				opts := map[string]any{}
				if tc.thinking != nil {
					opts["thinking"] = tc.thinking
				}
				params := provider.GenerateParams{ProviderOptions: opts}
				if streaming {
					result, err := model.DoStream(t.Context(), params)
					if err != nil {
						t.Fatal(err)
					}
					for c := range result.Stream {
						if c.Error != nil {
							t.Fatal(c.Error)
						}
					}
				} else {
					if _, err := model.DoGenerate(t.Context(), params); err != nil {
						t.Fatal(err)
					}
				}
				after, _ := json.Marshal(tc.thinking)
				if string(before) != string(after) {
					t.Fatal("caller input mutated")
				}
			})
		}
	}
}

func TestInputTransformationsAtStreamEOF(t *testing.T) {
	// Preserve diagnostics on the existing compatibility EOF finish path too.
	out := make(chan provider.StreamChunk, 4)
	go parseSSE(t.Context(), strings.NewReader("data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"input_transformations\":[{\"type\":\"thinking_dropped\",\"path\":\"messages.1.content.0\",\"reason\":\"prefix_binding_mismatch\"}]}}\n\n"), out, false)
	var found bool
	for chunk := range out {
		if chunk.Type == provider.ChunkFinish {
			found = chunk.Metadata["inputTransformations"] != nil
		}
	}
	if !found {
		t.Fatal("EOF discarded diagnostics")
	}
}

func TestThinkingBindingEmptyBetaHeader(t *testing.T) {
	for _, binding := range []bool{false, true} {
		t.Run(fmt.Sprint(binding), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				want := ""
				if binding {
					want = "thinking-binding-controls-2026-08-01"
				}
				if got := r.Header.Get("anthropic-beta"); got != want {
					t.Errorf("beta=%q want %q", got, want)
				}
				fmt.Fprint(w, `{"id":"m","content":[],"stop_reason":"end_turn","usage":{}}`)
			}))
			defer server.Close()
			model := Chat("claude-fable-5-1", WithAPIKey("test"), WithBaseURL(server.URL), WithAutoStreaming(false))
			opts := map[string]any{}
			if binding {
				opts["thinking"] = map[string]any{"blockBinding": map[string]any{"prefixMismatchBehavior": "error"}}
			}
			if _, err := model.DoGenerate(t.Context(), provider.GenerateParams{ProviderOptions: opts, Headers: map[string]string{"anthropic-beta": ""}}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestInputTransformations(t *testing.T) {
	const dropped = `[{"type":"thinking_dropped","reason":"prefix_binding_mismatch","path":"messages.1.content.0","extra":{"preserved":true}},{"type":"thinking_dropped","reason":"model_binding_mismatch","path":"messages.3.content.0"}]`
	for _, transport := range []string{"json", "auto-stream", "stream"} {
		for _, location := range []string{"start", "delta", "delta-only", "null-entry", "empty", "clear", "absent", "null", "null-delta"} {
			t.Run(transport+"/"+location, func(t *testing.T) {
				wantRaw := dropped
				if location == "null-entry" {
					wantRaw = `[null,{"type":"future","path":"messages.1.content.0","reason":"unknown"}]`
				}
				if location == "empty" || location == "clear" {
					wantRaw = "[]"
				}
				if location == "null" {
					wantRaw = "null"
				}
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body map[string]any
					json.NewDecoder(r.Body).Decode(&body)
					field := `,"input_transformations":` + wantRaw
					if location == "absent" {
						field = ""
					}
					if body["stream"] != true {
						fmt.Fprintf(w, `{"id":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{}%s}`, field)
						return
					}
					w.Header().Set("Content-Type", "text/event-stream")
					start, delta := field, ""
					if location == "delta" {
						start = `,"input_transformations":[]`
						delta = field
					}
					if location == "delta-only" {
						start, delta = "", field
					}
					if location == "clear" {
						start = `,"input_transformations":` + dropped
						delta = field
					}
					if location == "null-delta" {
						delta = `,"input_transformations":null`
					}
					fmt.Fprintf(w, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"content\":[],\"usage\":{}%s}}\n\n", start)
					fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
					fmt.Fprintf(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}%s}\n\ndata: {\"type\":\"message_stop\"}\n\n", delta)
				}))
				defer srv.Close()
				model := Chat("claude-fable-5-1", WithAPIKey("test"), WithBaseURL(srv.URL), WithAutoStreaming(transport == "auto-stream"))
				var result *goai.TextResult
				if transport == "stream" {
					stream, err := goai.StreamText(t.Context(), model, goai.WithPrompt("hi"))
					if err != nil {
						t.Fatal(err)
					}
					result = stream.Result()
					if err := stream.Err(); err != nil {
						t.Fatal(err)
					}
				} else {
					var err error
					result, err = goai.GenerateText(t.Context(), model, goai.WithPrompt("hi"))
					if err != nil {
						t.Fatal(err)
					}
				}
				got, exists := result.ProviderMetadata["anthropic"]["inputTransformations"]
				if location == "absent" || location == "null" {
					if exists {
						t.Fatal("invented transformations")
					}
					return
				}
				if !exists {
					t.Fatal("missing inputTransformations")
				}
				if _, ok := got.([]map[string]any); !ok {
					t.Errorf("inputTransformations type = %T, want []map[string]any", got)
				}
				if _, ok := result.Response.ProviderMetadata["inputTransformations"].([]map[string]any); !ok {
					t.Errorf("Response inputTransformations type = %T", result.Response.ProviderMetadata["inputTransformations"])
				}
				encoded, err := json.Marshal(got)
				if err != nil {
					t.Fatal(err)
				}
				var actual, want any
				json.Unmarshal(encoded, &actual)
				json.Unmarshal([]byte(wantRaw), &want)
				if !reflect.DeepEqual(actual, want) {
					t.Fatalf("transformations=%s want %s", encoded, wantRaw)
				}
				responseEncoded, _ := json.Marshal(result.Response.ProviderMetadata["inputTransformations"])
				if string(responseEncoded) != string(encoded) {
					t.Fatal("Response.ProviderMetadata lost diagnostics")
				}
			})
		}
	}
}

func TestInputTransformationsMalformedStream(t *testing.T) {
	for _, event := range []string{"message_start", "message_delta"} {
		for _, value := range []string{`{}`, `[1]`} {
			t.Run(event+value, func(t *testing.T) {
				data := fmt.Sprintf(`{"type":"message_delta","input_transformations":%s}`, value)
				if event == "message_start" {
					data = fmt.Sprintf(`{"type":"message_start","message":{"input_transformations":%s}}`, value)
				}
				out := make(chan provider.StreamChunk, 4)
				go parseSSE(t.Context(), strings.NewReader("data: "+data+"\n\n"), out, false)
				var failed bool
				for chunk := range out {
					if chunk.Error != nil && strings.Contains(chunk.Error.Error(), "input_transformations") {
						failed = true
					}
					if chunk.Type == provider.ChunkFinish {
						t.Fatal("invalid diagnostic accepted")
					}
				}
				if !failed {
					t.Fatal("missing diagnostic decode error")
				}
			})
		}
	}
}
