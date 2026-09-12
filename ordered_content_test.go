package goai

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

// Provider integration tests in other packages do not instrument goai itself
// under CI's default coverprofile command. Exercise the replay contract here too.
func TestOrderedContentReplay(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, loop := range []bool{false, true} {
			t.Run(fmt.Sprintf("stream=%v/loop=%v", streaming, loop), func(t *testing.T) {
				first := []provider.Part{
					{Type: provider.PartText, Text: "before", ProviderOptions: map[string]any{"itemId": "m1", "phase": "commentary"}},
					{Type: provider.PartToolCall, ToolCallID: "c", ToolName: "lookup", ToolInput: json.RawMessage(`{}`)},
					{Type: provider.PartText, Text: "after", ProviderOptions: map[string]any{"itemId": "m2", "phase": "final_answer"}},
				}
				final := []provider.Part{{Type: provider.PartText, Text: "answer", ProviderOptions: map[string]any{"itemId": "m3", "phase": "final_answer"}}}
				calls := 0
				generate := func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
					calls++
					if calls == 1 {
						return &provider.GenerateResult{Text: "beforeafter", Content: first, ToolCalls: []provider.ToolCall{{ID: "c", Name: "lookup", Input: json.RawMessage(`{}`)}}, FinishReason: provider.FinishToolCalls}, nil
					}
					if len(p.Messages) != 3 || !reflect.DeepEqual(p.Messages[1].Content, first) {
						t.Errorf("tool continuation lost ordered Content: %#v", p.Messages)
					}
					return &provider.GenerateResult{Text: "answer", Content: final, FinishReason: provider.FinishStop}, nil
				}
				model := &mockModel{id: "ordered", generateFn: generate, streamFn: func(ctx context.Context, p provider.GenerateParams) (*provider.StreamResult, error) {
					r, err := generate(ctx, p)
					if err != nil {
						return nil, err
					}
					chunks := []provider.StreamChunk{{Type: provider.ChunkText, Text: r.Text}}
					for _, tc := range r.ToolCalls {
						chunks = append(chunks, provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: tc.ID, ToolName: tc.Name, ToolInput: string(tc.Input)})
					}
					chunks = append(chunks, provider.StreamChunk{Type: provider.ChunkFinish, Content: r.Content, FinishReason: r.FinishReason})
					return streamFromChunks(chunks...), nil
				}}
				var hooks []StepResult
				opts := []Option{WithPrompt("question"), WithOnStepFinish(func(s StepResult) { hooks = append(hooks, s) })}
				if loop {
					opts = append(opts, WithMaxSteps(2), WithTools(Tool{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`), Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}))
				}
				var result *TextResult
				var err error
				if streaming {
					stream, e := StreamText(t.Context(), model, opts...)
					if e != nil {
						t.Fatal(e)
					}
					result = stream.Result()
					err = stream.Err()
				} else {
					result, err = GenerateText(t.Context(), model, opts...)
				}
				if err != nil {
					t.Fatal(err)
				}
				wantSteps := 1
				if loop {
					wantSteps = 2
				}
				if len(hooks) != wantSteps || len(result.Steps) != wantSteps {
					t.Fatalf("hooks=%d steps=%d", len(hooks), len(result.Steps))
				}
				if !reflect.DeepEqual(result.ResponseMessages[0].Content, first) {
					t.Fatalf("replay=%#v", result.ResponseMessages)
				}
				for i, hook := range hooks {
					if !reflect.DeepEqual(hook.Content, result.Steps[i].Content) {
						t.Fatalf("step %d hook lost Content", i)
					}
				}
				if loop && !reflect.DeepEqual(result.ResponseMessages[2].Content, final) {
					t.Fatal("last step Content lost")
				}
				// The SDK's reconstructed history owns its outer option maps and tool bytes.
				result.ResponseMessages[0].Content[0].ProviderOptions["phase"] = "changed"
				result.ResponseMessages[0].Content[1].ToolInput[0] = 'x'
				if first[0].ProviderOptions["phase"] != "commentary" || string(first[1].ToolInput) != "{}" {
					t.Fatal("replay mutations reached provider output")
				}
			})
		}
	}
}

func TestOrderedContentEmptySnapshot(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content []provider.Part
		empty   bool
	}{{"nil-falls-back", nil, false}, {"empty-is-authoritative", []provider.Part{}, true}} {
		t.Run(tc.name, func(t *testing.T) {
			model := &mockModel{id: "ordered", generateFn: func(context.Context, provider.GenerateParams) (*provider.GenerateResult, error) {
				return &provider.GenerateResult{Text: "fallback", Content: tc.content, FinishReason: provider.FinishStop}, nil
			}, streamFn: func(context.Context, provider.GenerateParams) (*provider.StreamResult, error) {
				return streamFromChunks(provider.StreamChunk{Type: provider.ChunkText, Text: "fallback"}, provider.StreamChunk{Type: provider.ChunkFinish, Content: tc.content, FinishReason: provider.FinishStop}), nil
			}}
			sync, err := GenerateText(t.Context(), model, WithPrompt("question"))
			if err != nil {
				t.Fatal(err)
			}
			stream, err := StreamText(t.Context(), model, WithPrompt("question"))
			if err != nil {
				t.Fatal(err)
			}
			async := stream.Result()
			if err := stream.Err(); err != nil {
				t.Fatal(err)
			}
			for _, r := range []*TextResult{sync, async} {
				if tc.empty {
					if r.ResponseMessages != nil {
						t.Fatal("empty snapshot fell back to aggregate text")
					}
				} else if len(r.ResponseMessages) != 1 || r.ResponseMessages[0].Content[0].Text != "fallback" {
					t.Fatalf("nil snapshot broke fallback: %#v", r.ResponseMessages)
				}
			}
		})
	}
}
