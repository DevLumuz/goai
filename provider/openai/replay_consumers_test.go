package openai

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

func replayMessage(id, phase, text string) map[string]any {
	return map[string]any{"type": "message", "id": id, "role": "assistant", "phase": phase, "content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}}}
}

func emitReplayResponse(w http.ResponseWriter, output []map[string]any, streaming, done bool) {
	if !streaming {
		json.NewEncoder(w).Encode(map[string]any{"id": "r", "model": "gpt-5", "status": "completed", "output": output})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	emit := func(v any) { data, _ := json.Marshal(v); fmt.Fprintf(w, "data: %s\n\n", data) }
	for i, item := range output {
		emit(map[string]any{"type": "response.output_item.added", "output_index": i, "item": item})
		if item["type"] == "message" {
			for _, raw := range item["content"].([]any) {
				c := raw.(map[string]any)
				event, key := "response.output_text.delta", "text"
				if c["type"] == "refusal" {
					event, key = "response.refusal.delta", "refusal"
				}
				emit(map[string]any{"type": event, "item_id": item["id"], "output_index": i, "delta": c[key]})
			}
		} else if item["type"] == "function_call" {
			emit(map[string]any{"type": "response.function_call_arguments.delta", "output_index": i, "delta": item["arguments"]})
			emit(map[string]any{"type": "response.function_call_arguments.done", "output_index": i})
		}
		emit(map[string]any{"type": "response.output_item.done", "output_index": i, "item": item})
	}
	if done {
		fmt.Fprint(w, "data: [DONE]\n\n")
	} else {
		emit(map[string]any{"type": "response.completed", "response": map[string]any{"id": "r", "model": "gpt-5"}})
	}
}

func TestResponsesObjectContentConsumers(t *testing.T) {
	type answer struct {
		Value int `json:"value"`
	}
	for _, mode := range []string{"generate", "generate-tools", "stream", "stream-partials"} {
		t.Run(mode, func(t *testing.T) {
			requests := make(chan map[string]any, 4)
			final := []map[string]any{{"type": "reasoning", "id": "reason", "encrypted_content": "opaque"}, replayMessage("final", "final_answer", `{"value":42}`)}
			first := []map[string]any{replayMessage("comment", "commentary", "checking"), {"type": "function_call", "id": "fc", "call_id": "c", "name": "lookup", "arguments": "{}"}}
			n := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				json.NewDecoder(r.Body).Decode(&body)
				requests <- body
				n++
				output := final
				if mode == "generate-tools" && n == 1 {
					output = first
				}
				emitReplayResponse(w, output, body["stream"] == true, false)
			}))
			defer server.Close()
			model := Chat("gpt-5", WithAPIKey("test"), WithBaseURL(server.URL))
			var hooks []goai.StepResult
			opts := []goai.Option{goai.WithPrompt("answer"), goai.WithProviderOptions(map[string]any{"store": false}), goai.WithOnStepFinish(func(s goai.StepResult) { hooks = append(hooks, s) })}
			if mode == "generate-tools" {
				opts = append(opts, goai.WithMaxSteps(2), goai.WithTools(goai.Tool{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`), Execute: func(context.Context, json.RawMessage) (string, error) { return "42", nil }}))
			}
			var result *goai.ObjectResult[answer]
			var err error
			if mode == "stream" || mode == "stream-partials" {
				stream, e := goai.StreamObject[answer](t.Context(), model, opts...)
				if e != nil {
					t.Fatal(e)
				}
				if mode == "stream-partials" {
					for range stream.PartialObjectStream() {
					}
				}
				result, err = stream.Result()
			} else {
				result, err = goai.GenerateObject[answer](t.Context(), model, opts...)
			}
			if err != nil {
				t.Fatal(err)
			}
			if result.Object.Value != 42 {
				t.Fatalf("object=%v", result.Object)
			}
			wantSteps := 1
			if mode == "generate-tools" {
				wantSteps = 2
			}
			if len(hooks) != wantSteps {
				t.Fatalf("got %d hooks, want %d", len(hooks), wantSteps)
			}
			for _, s := range hooks {
				if len(s.Content) == 0 {
					t.Errorf("step %d hook lost Content", s.Number)
				}
			}
			for _, s := range result.Steps {
				if len(s.Content) == 0 {
					t.Errorf("step %d lost Content", s.Number)
				}
			}
			lastParts := result.ResponseMessages[len(result.ResponseMessages)-1].Content
			if len(lastParts) != 2 || lastParts[0].Type != provider.PartReasoning {
				t.Fatalf("ordered reasoning lost: %#v", lastParts)
			}
			<-requests
			if mode == "generate-tools" {
				body := <-requests
				input := body["input"].([]any)
				var commentary bool
				for _, v := range input {
					m := v.(map[string]any)
					if m["phase"] == "commentary" {
						commentary = true
					}
				}
				if !commentary {
					t.Error("tool continuation lost commentary phase")
				}
			}
			saved, _ := json.Marshal(result.ResponseMessages)
			var messages []provider.Message
			json.Unmarshal(saved, &messages)
			if _, err := model.DoGenerate(t.Context(), provider.GenerateParams{Messages: messages}); err != nil {
				t.Fatal(err)
			}
			input := (<-requests)["input"].([]any)
			last := input[len(input)-1].(map[string]any)
			if last["phase"] != "final_answer" {
				t.Fatalf("replay lost final phase: %#v", input)
			}
		})
	}
}

func TestResponsesStreamContentConsumers(t *testing.T) {
	for _, done := range []bool{false, true} {
		for _, loop := range []bool{false, true} {
			t.Run(fmt.Sprintf("done=%v/loop=%v", done, loop), func(t *testing.T) {
				n := 0
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					n++
					output := []map[string]any{replayMessage("final", "final_answer", "answer")}
					if loop && n == 1 {
						output = []map[string]any{replayMessage("comment", "commentary", "checking"), {"type": "function_call", "id": "fc", "call_id": "c", "name": "lookup", "arguments": "{}"}}
					}
					emitReplayResponse(w, output, true, done)
				}))
				defer server.Close()
				model := Chat("gpt-5", WithAPIKey("test"), WithBaseURL(server.URL), WithResponsesStreamDoneCompatibility(done))
				var hooks []goai.StepResult
				opts := []goai.Option{goai.WithPrompt("answer"), goai.WithProviderOptions(map[string]any{"store": false}), goai.WithOnStepFinish(func(s goai.StepResult) { hooks = append(hooks, s) })}
				if loop {
					opts = append(opts, goai.WithMaxSteps(2), goai.WithTools(goai.Tool{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`), Execute: func(context.Context, json.RawMessage) (string, error) { return "42", nil }}))
				}
				stream, err := goai.StreamText(t.Context(), model, opts...)
				if err != nil {
					t.Fatal(err)
				}
				var final provider.StreamChunk
				var stepContents [][]provider.Part
				for c := range stream.Stream() {
					if c.Type == provider.ChunkFinish {
						final = c
					}
					if c.Type == provider.ChunkStepFinish && c.Metadata["stepSource"] == "goai" {
						stepContents = append(stepContents, c.Content)
					}
				}
				if err := stream.Err(); err != nil {
					t.Fatal(err)
				}
				result := stream.Result()
				wantSteps := 1
				if loop {
					wantSteps = 2
				}
				if len(hooks) != wantSteps || len(result.Steps) != wantSteps {
					t.Fatalf("hooks=%d steps=%d, want %d", len(hooks), len(result.Steps), wantSteps)
				}
				if len(final.Content) != 1 || final.Content[0].ProviderOptions["phase"] != "final_answer" {
					t.Errorf("final snapshot=%#v", final.Content)
				}
				for i, s := range hooks {
					if len(s.Content) == 0 || !reflect.DeepEqual(s.Content, result.Steps[i].Content) {
						t.Errorf("hook %d snapshot=%#v", i, s.Content)
					}
				}
				if loop {
					if len(stepContents) != 2 || len(stepContents[0]) != 2 || !reflect.DeepEqual(stepContents[1], final.Content) {
						t.Errorf("step snapshots=%#v", stepContents)
					}
				}
				last := result.ResponseMessages[len(result.ResponseMessages)-1]
				if last.Content[0].ProviderOptions["phase"] != "final_answer" {
					t.Error("ResponseMessages lost phase")
				}
			})
		}
	}
}

func TestResponsesRefusalTextParity(t *testing.T) {
	for _, mixed := range []bool{false, true} {
		t.Run(fmt.Sprint(mixed), func(t *testing.T) {
			parts := []any{map[string]any{"type": "refusal", "refusal": "Cannot comply"}}
			want := "Cannot comply"
			if mixed {
				parts = append([]any{map[string]any{"type": "output_text", "text": "Notice: "}}, parts...)
				want = "Notice: Cannot comply"
			}
			output := []map[string]any{{"type": "message", "id": "m", "phase": "final_answer", "role": "assistant", "content": parts}}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				json.NewDecoder(r.Body).Decode(&body)
				emitReplayResponse(w, output, body["stream"] == true, false)
			}))
			defer server.Close()
			model := Chat("gpt-5", WithAPIKey("test"), WithBaseURL(server.URL))
			sync, err := goai.GenerateText(t.Context(), model, goai.WithPrompt("hi"))
			if err != nil {
				t.Fatal(err)
			}
			stream, err := goai.StreamText(t.Context(), model, goai.WithPrompt("hi"))
			if err != nil {
				t.Fatal(err)
			}
			async := stream.Result()
			if err := stream.Err(); err != nil {
				t.Fatal(err)
			}
			if sync.Text != want || async.Text != want {
				t.Errorf("sync=%q stream=%q want=%q", sync.Text, async.Text, want)
			}
			if !reflect.DeepEqual(sync.ResponseMessages, async.ResponseMessages) {
				t.Error("refusal replay differs by transport")
			}
			wire := convertToResponsesInput(sync.ResponseMessages)
			content := wire[0]["content"].([]map[string]any)
			if content[len(content)-1]["type"] != "refusal" {
				t.Error("refusal wire type lost")
			}
		})
	}
}
