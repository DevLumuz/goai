package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

// Run the real codec and HTTP boundary in its owning package so CI attributes
// this coverage to openaicompat, not only to the OpenRouter wrapper's tests.
func TestReasoningDetailsHTTPReplay(t *testing.T) {
	for _, tc := range []struct{ name, details, text, extra string }{
		{"visible", `[{"type":"reasoning.text","text":"think","signature":"sig","id":"r"},{"type":"reasoning.summary","summary":" summary"}]`, "think summary", ""},
		{"encrypted", `[{"type":"reasoning.encrypted","data":"opaque","id":"e","extra":{"keep":true}}]`, "", ""},
		{"empty", `[]`, "", ""},
		{"with-usage", `[{"type":"reasoning.text","text":"think"}]`, "think", `,"usage":{"completion_tokens_details":{"accepted_prediction_tokens":2}}`},
		{"with-citations", `[{"type":"reasoning.text","text":"think"}]`, "think", `,"citations":["https://example.com"]`},
	} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%v", tc.name, streaming), func(t *testing.T) {
				requests := make(chan map[string]any, 2)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
						return
					}
					requests <- body
					if body["stream"] == true {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"answer\",\"reasoning_details\":%s}}]%s}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", tc.details, tc.extra)
					} else {
						fmt.Fprintf(w, `{"choices":[{"message":{"content":"answer","reasoning_details":%s},"finish_reason":"stop"}]%s}`, tc.details, tc.extra)
					}
				}))
				defer server.Close()
				model := NewChatModel(ChatModelConfig{ProviderID: "openrouter", ModelID: "reasoner", BaseURL: server.URL, TokenSource: provider.StaticToken("test"), RequestConfig: RequestConfig{IncludeReasoningDetails: true}})
				var result *goai.TextResult
				var err error
				if streaming {
					stream, e := goai.StreamText(t.Context(), model, goai.WithPrompt("think"))
					if e != nil {
						t.Fatal(e)
					}
					result = stream.Result()
					err = stream.Err()
				} else {
					result, err = goai.GenerateText(t.Context(), model, goai.WithPrompt("think"))
				}
				if err != nil {
					t.Fatal(err)
				}
				if result.Reasoning != tc.text {
					t.Fatalf("reasoning=%q want %q", result.Reasoning, tc.text)
				}
				meta := result.ProviderMetadata["openrouter"]["reasoning_details"]
				gotJSON, err := json.Marshal(meta)
				if err != nil {
					t.Fatal(err)
				}
				var want, got any
				json.Unmarshal([]byte(tc.details), &want)
				json.Unmarshal(gotJSON, &got)
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("metadata=%s want %s", gotJSON, tc.details)
				}
				if tc.name == "with-usage" && result.ProviderMetadata["openai"]["acceptedPredictionTokens"] == nil {
					t.Fatal("reasoning metadata replaced existing usage metadata")
				}
				if tc.name == "with-citations" && len(result.Sources) != 1 {
					t.Fatal("reasoning metadata replaced citations")
				}
				<-requests
				persisted, _ := json.Marshal(result.ResponseMessages)
				var messages []provider.Message
				if err := json.Unmarshal(persisted, &messages); err != nil {
					t.Fatal(err)
				}
				if _, err := model.DoGenerate(t.Context(), provider.GenerateParams{Messages: messages}); err != nil {
					t.Fatal(err)
				}
				body := <-requests
				assistant := body["messages"].([]any)[0].(map[string]any)
				if !reflect.DeepEqual(assistant["reasoning_details"], want) {
					t.Fatalf("replay=%#v want %s", assistant, tc.details)
				}
				if assistant["reasoning"] != nil || assistant["reasoning_content"] != nil {
					t.Fatal("structured replay duplicated as plaintext")
				}
			})
		}
	}
}

func TestReasoningDetailsCancellation(t *testing.T) {
	for _, tc := range []struct{ name, detail string }{{"visible-delta", `[{"type":"reasoning.text","text":"think"}]`}, {"final-metadata", `[{"type":"reasoning.encrypted","data":"opaque"}]`}} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			out := make(chan provider.StreamChunk)
			stopped := make(chan struct{})
			go func() {
				defer close(stopped)
				ParseStream(ctx, sse.NewScanner(strings.NewReader(fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"reasoning_details\":%s}}]}\n\ndata: [DONE]\n\n", tc.detail))), out)
			}()
			// Do not receive until the producer exits: cancellation, rather than a
			// ready receiver chosen at random by select, must unblock the send.
			<-stopped
			if _, ok := <-out; ok {
				t.Fatal("cancelled stream emitted a chunk")
			}
		})
	}
}
