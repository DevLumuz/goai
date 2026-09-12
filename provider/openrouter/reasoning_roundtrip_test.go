package openrouter

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

func TestReasoningRoundTrip(t *testing.T) {
	cases := []struct {
		name, fields, want, text string
		deltas                   []string
	}{
		{name: "string", fields: `"reasoning":"think"`, want: `{"reasoning":"think"}`, text: "think", deltas: []string{`"reasoning":"thi"`, `"reasoning":"nk"`}},
		{name: "alias", fields: `"reasoning_content":"think"`, want: `{"reasoning":"think"}`, text: "think", deltas: []string{`"reasoning_content":"think"`}},
		{name: "empty", fields: `"reasoning_details":[]`, want: `{"reasoning_details":[]}`, deltas: []string{`"reasoning_details":[]`}},
		{name: "encrypted-only", fields: `"reasoning_details":[{"type":"reasoning.encrypted","data":"opaque","id":"e1"}]`, want: `{"reasoning_details":[{"type":"reasoning.encrypted","data":"opaque","id":"e1"}]}`, deltas: []string{`"reasoning_details":[{"type":"reasoning.encrypted","data":"opaque","id":"e1"}]`}},
		{name: "blocks", text: "thinksummary", fields: `"reasoning":"thinksummary","reasoning_details":[{"type":"reasoning.text","text":"think","signature":"sig","id":"r1","index":0,"format":"anthropic-claude-v1","extra":{"v":1}},{"type":"reasoning.summary","summary":"summary","index":0},{"type":"reasoning.encrypted","data":"a","id":"e1","index":0},{"type":"reasoning.encrypted","data":"b","id":"e2","index":0}]`,
			want: `{"reasoning_details":[{"type":"reasoning.text","text":"think","signature":"sig","id":"r1","index":0,"format":"anthropic-claude-v1","extra":{"v":1}},{"type":"reasoning.summary","summary":"summary","index":0},{"type":"reasoning.encrypted","data":"a","id":"e1","index":0},{"type":"reasoning.encrypted","data":"b","id":"e2","index":0}]}`,
			deltas: []string{
				`"reasoning":"thi","reasoning_details":[{"type":"reasoning.text","text":"thi","signature":null,"id":"r1","index":0,"format":"anthropic-claude-v1","extra":{"v":1}}]`,
				`"reasoning_details":[{"type":"reasoning.text","text":"nk","id":"r1","index":0}]`,
				`"content":"answer","reasoning_details":[{"type":"reasoning.text","signature":"sig","id":"r1","index":0},{"type":"reasoning.summary","summary":"sum","index":0}]`,
				`"reasoning_details":[{"type":"reasoning.summary","summary":"mary","index":0},{"type":"reasoning.encrypted","data":"a","id":"e1","index":0},{"type":"reasoning.encrypted","data":"b","id":"e2","index":0}]`,
			}},
	}
	for _, tc := range cases {
		for _, streaming := range []bool{false, true} {
			for _, loop := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/stream=%v/loop=%v", tc.name, streaming, loop), func(t *testing.T) {
					requests := make(chan map[string]any, 4)
					n := 0
					srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						var body map[string]any
						if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
							t.Error(err)
							return
						}
						requests <- body
						n++
						if n > 1 {
							if body["stream"] == true {
								w.Header().Set("Content-Type", "text/event-stream")
								fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
							} else {
								json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "ok"}, "finish_reason": "stop"}}})
							}
							return
						}
						if body["stream"] == true {
							w.Header().Set("Content-Type", "text/event-stream")
							for _, delta := range tc.deltas {
								fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{%s}}]}\n\n", delta)
							}
							if loop {
								fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n")
							} else {
								fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
							}
							fmt.Fprint(w, "data: [DONE]\n\n")
						} else {
							extra, finish := "", "stop"
							if loop {
								extra = `,"tool_calls":[{"id":"c1","function":{"name":"lookup","arguments":"{}"}}]`
								finish = "tool_calls"
							}
							fmt.Fprintf(w, `{"choices":[{"message":{%s%s},"finish_reason":%q}],"usage":{"completion_tokens_details":{"reasoning_tokens":3}}}`, tc.fields, extra, finish)
						}
					}))
					defer srv.Close()
					model := Chat("reasoner", WithAPIKey("test"), WithBaseURL(srv.URL))
					opts := []goai.Option{goai.WithPrompt("think")}
					if loop {
						opts = append(opts, goai.WithMaxSteps(2), goai.WithTools(goai.Tool{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`), Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}))
					}
					var result *goai.TextResult
					if streaming {
						stream, err := goai.StreamText(t.Context(), model, opts...)
						if err != nil {
							t.Fatal(err)
						}
						result = stream.Result()
						if err := stream.Err(); err != nil {
							t.Fatal(err)
						}
					} else {
						var err error
						result, err = goai.GenerateText(t.Context(), model, opts...)
						if err != nil {
							t.Fatal(err)
						}
					}
					if result.Reasoning != tc.text {
						t.Fatalf("reasoning=%q want %q", result.Reasoning, tc.text)
					}
					<-requests
					var replay map[string]any
					if loop {
						replay = <-requests
					} else {
						data, err := json.Marshal(result.ResponseMessages)
						if err != nil {
							t.Fatal(err)
						}
						var messages []provider.Message
						if err := json.Unmarshal(data, &messages); err != nil {
							t.Fatal(err)
						}
						if _, err := model.DoGenerate(t.Context(), provider.GenerateParams{Messages: messages}); err != nil {
							t.Fatal(err)
						}
						replay = <-requests
					}
					var assistant map[string]any
					for _, item := range replay["messages"].([]any) {
						m := item.(map[string]any)
						if m["role"] == "assistant" {
							assistant = m
							break
						}
					}
					if assistant == nil {
						t.Fatal("assistant history lost")
					}
					got := map[string]any{}
					for _, k := range []string{"reasoning", "reasoning_content", "reasoning_details"} {
						if v, ok := assistant[k]; ok {
							got[k] = v
						}
					}
					var want map[string]any
					json.Unmarshal([]byte(tc.want), &want)
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("replay=%#v want %s", got, tc.want)
					}
				})
			}
		}
	}
}
