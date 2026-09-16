package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// Exercise the public SDK and a second real HTTP request, not just the codec.
func TestResponsesPhaseRoundTrip(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, toolLoop := range []bool{false, true} {
			t.Run(fmt.Sprintf("stream=%v/loop=%v", streaming, toolLoop), func(t *testing.T) {
				items := []map[string]any{
					{"type": "message", "id": "msg_c1", "role": "assistant", "phase": "commentary", "content": []any{map[string]any{"type": "output_text", "text": "Checking", "annotations": []any{}}}},
					{"type": "message", "id": "msg_c2", "role": "assistant", "phase": "commentary", "content": []any{map[string]any{"type": "output_text", "text": " now", "annotations": []any{}}}},
					{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
					{"type": "message", "id": "msg_f", "role": "assistant", "phase": "final_answer", "content": []any{map[string]any{"type": "output_text", "text": "Done", "annotations": []any{}}}},
				}
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
					output := items
					if n > 1 {
						output = items[3:]
					}
					if body["stream"] == true {
						w.Header().Set("Content-Type", "text/event-stream")
						emit := func(event map[string]any) { data, _ := json.Marshal(event); fmt.Fprintf(w, "data: %s\n\n", data) }
						for i, item := range output {
							emit(map[string]any{"type": "response.output_item.added", "output_index": i, "item": item})
							if item["type"] == "message" {
								emit(map[string]any{"type": "response.output_text.delta", "output_index": i, "item_id": item["id"], "delta": item["content"].([]any)[0].(map[string]any)["text"]})
							} else {
								emit(map[string]any{"type": "response.function_call_arguments.delta", "output_index": i, "delta": "{}"})
								emit(map[string]any{"type": "response.function_call_arguments.done", "output_index": i})
							}
							emit(map[string]any{"type": "response.output_item.done", "output_index": i, "item": item})
						}
						// Omit output to exercise item.done assembly as well as non-stream parse.
						emit(map[string]any{"type": "response.completed", "response": map[string]any{"id": "resp_1", "model": "gpt-5"}})
					} else {
						json.NewEncoder(w).Encode(map[string]any{"id": "resp_1", "model": "gpt-5", "status": "completed", "output": output})
					}
				}))
				defer srv.Close()
				model := Chat("gpt-5", WithAPIKey("test"), WithBaseURL(srv.URL))
				opts := []goai.Option{goai.WithPrompt("check"), goai.WithProviderOptions(map[string]any{"useResponsesAPI": true, "store": false})}
				if toolLoop {
					opts = append(opts, goai.WithMaxSteps(2), goai.WithTools(goai.Tool{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`), Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}))
				}
				var result *goai.TextResult
				var err error
				if streaming {
					stream, streamErr := goai.StreamText(t.Context(), model, opts...)
					if streamErr != nil {
						t.Fatal(streamErr)
					}
					result, err = stream.Result(), stream.Err()
				} else {
					result, err = goai.GenerateText(t.Context(), model, opts...)
				}
				if err != nil {
					t.Fatal(err)
				}
				<-requests
				var replay map[string]any
				if toolLoop {
					replay = <-requests
				} else {
					// JSON persistence must not change the replay representation.
					saved, err := json.Marshal(result.ResponseMessages)
					if err != nil {
						t.Fatal(err)
					}
					var messages []provider.Message
					if err := json.Unmarshal(saved, &messages); err != nil {
						t.Fatal(err)
					}
					_, err = model.DoGenerate(t.Context(), provider.GenerateParams{Messages: messages, ProviderOptions: map[string]any{"useResponsesAPI": true, "store": false}})
					if err != nil {
						t.Fatal(err)
					}
					replay = <-requests
				}
				input := replay["input"].([]any)
				var got []any
				for _, raw := range input {
					item := raw.(map[string]any)
					if item["role"] == "assistant" || item["type"] == "function_call" {
						got = append(got, raw)
					}
				}
				var want []any
				for _, item := range items {
					copy := make(map[string]any)
					for k, v := range item {
						if k != "id" {
							copy[k] = v
						}
					}
					want = append(want, copy)
				}
				wantJSON, _ := json.Marshal(want)
				var normalized any
				json.Unmarshal(wantJSON, &normalized)
				if !reflect.DeepEqual(got, normalized) {
					t.Fatalf("replay = %#v, want %s", got, wantJSON)
				}
			})
		}
	}
}

func TestResponsesPhaseStreamSources(t *testing.T) {
	for _, source := range []string{"added", "done", "terminal", "absent"} {
		t.Run(source, func(t *testing.T) {
			addedPhase, donePhase, output := "", "", ""
			if source == "added" {
				addedPhase = `,"phase":"commentary"`
			}
			if source == "done" {
				donePhase = `,"phase":"commentary"`
			}
			if source == "terminal" {
				output = `,"output":[{"type":"message","id":"m","phase":"commentary","content":[{"type":"output_text","text":"final answer"}]}]`
			}
			wire := fmt.Sprintf("data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"m\"%s}}\n\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"item_id\":\"m\",\"delta\":\"final answer\"}\n\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"m\",\"content\":[{\"type\":\"output_text\",\"text\":\"final answer\"}]%s}}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\"%s}}\n\n", addedPhase, donePhase, output)
			out := make(chan provider.StreamChunk, 8)
			go streamResponses(t.Context(), io.NopCloser(strings.NewReader(wire)), out)
			var content []provider.Part
			for chunk := range out {
				if chunk.Error != nil {
					t.Fatal(chunk.Error)
				}
				if chunk.Type == provider.ChunkText && source == "added" && chunk.Metadata["phase"] != "commentary" {
					t.Fatal("missing live phase")
				}
				if chunk.Type == provider.ChunkFinish {
					content = chunk.Content
				}
			}
			if len(content) != 1 {
				t.Fatalf("content=%#v", content)
			}
			phase := content[0].ProviderOptions["phase"]
			if source == "absent" {
				if phase != nil {
					t.Fatalf("guessed phase=%v", phase)
				}
			} else if phase != "commentary" {
				t.Fatalf("phase=%v", phase)
			}
		})
	}
}

func TestResponsesReplaySizeLimit(t *testing.T) {
	old := maxResponsesReplayBytes
	maxResponsesReplayBytes = 10
	t.Cleanup(func() { maxResponsesReplayBytes = old })
	out := make(chan provider.StreamChunk, 4)
	go streamResponses(t.Context(), io.NopCloser(strings.NewReader("data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"m\"}}\n\n")), out)
	var failed bool
	for chunk := range out {
		if chunk.Error != nil && strings.Contains(chunk.Error.Error(), "size limit") {
			failed = true
		}
		if chunk.Type == provider.ChunkFinish {
			t.Fatal("oversized replay accepted")
		}
	}
	if !failed {
		t.Fatal("missing size limit error")
	}
}

func TestResponsesOrderedContentRetainsRefusal(t *testing.T) {
	result, err := parseResponsesResult([]byte(`{"output":[{"type":"reasoning","id":"r1","encrypted_content":"opaque"},{"type":"message","id":"m1","phase":"final_answer","content":[{"type":"refusal","refusal":"Cannot comply"}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	input := convertToResponsesInput([]provider.Message{{Role: provider.RoleAssistant, Content: result.Content}})
	if len(input) != 2 || input[1]["phase"] != "final_answer" {
		t.Fatalf("input=%#v", input)
	}
	content := input[1]["content"].([]map[string]any)
	if len(content) != 1 || content[0]["refusal"] != "Cannot comply" {
		t.Fatalf("refusal lost: %#v", content)
	}
}
