package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
)

func TestResponsesReplayRejectsInvalidItems(t *testing.T) {
	for _, raw := range []string{`{`, `{"type":"message","content":"not an array"}`} {
		t.Run(raw, func(t *testing.T) {
			content, err := responsesReplayContent([]json.RawMessage{json.RawMessage(raw)}, nil, nil)
			if err == nil || content != nil {
				t.Fatalf("invalid replay accepted: content=%#v err=%v", content, err)
			}
		})
	}
}

func TestResponsesStreamRejectsMalformedReplay(t *testing.T) {
	input := "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"content\":\"not an array\"}}\n\n" + completedResponsesEvent
	out := make(chan provider.StreamChunk, 4)
	go streamResponses(t.Context(), io.NopCloser(strings.NewReader(input)), out)
	var gotError bool
	for c := range out {
		if c.Type == provider.ChunkError {
			gotError = c.Error != nil
		}
		if c.Type == provider.ChunkFinish {
			t.Fatal("malformed replay reported as successful")
		}
	}
	if !gotError {
		t.Fatal("missing replay decode error")
	}
}

func TestResponsesTextMetadataWithoutAddedEvent(t *testing.T) {
	input := "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"message-only-delta\",\"delta\":\"hello\"}\n\n" + completedResponsesEvent
	out := make(chan provider.StreamChunk, 4)
	go streamResponses(t.Context(), io.NopCloser(strings.NewReader(input)), out)
	var seen bool
	for c := range out {
		if c.Error != nil {
			t.Fatal(c.Error)
		}
		if c.Type == provider.ChunkText {
			seen = true
			if c.Text != "hello" || c.Metadata["itemId"] != "message-only-delta" {
				t.Fatalf("chunk=%#v", c)
			}
			if _, ok := c.Metadata["phase"]; ok {
				t.Fatal("invented phase without an item event")
			}
		}
	}
	if !seen {
		t.Fatal("missing text delta")
	}
}

func TestResponsesFinishCancellation(t *testing.T) {
	for _, cancelAfter := range []int{1, 2} {
		t.Run(fmt.Sprintf("after-tool-%d", cancelAfter), func(t *testing.T) {
			var input strings.Builder
			for i := range 2 {
				fmt.Fprintf(&input, "data: {\"type\":\"response.output_item.added\",\"output_index\":%d,\"item\":{\"type\":\"function_call\",\"id\":\"f%d\",\"call_id\":\"c%d\",\"name\":\"lookup\"}}\n\n", i, i, i)
				fmt.Fprintf(&input, "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":%d,\"delta\":\"{}\"}\n\n", i)
			}
			input.WriteString(completedResponsesEvent)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			out := make(chan provider.StreamChunk)
			stopped := make(chan struct{})
			go func() {
				defer close(stopped)
				streamResponses(ctx, io.NopCloser(strings.NewReader(input.String())), out)
			}()
			calls := 0
			for c := range out {
				if c.Error != nil {
					t.Fatal(c.Error)
				}
				if c.Type == provider.ChunkToolCall {
					if c.ToolCallID != fmt.Sprintf("c%d", calls) || c.ToolInput != "{}" {
						t.Errorf("flushed call=%#v", c)
					}
					calls++
					if calls == cancelAfter {
						cancel()
						break
					}
				}
			}
			// Receiving a flushed call proves we are inside the terminal finalizer.
			// With no receiver for the next send, only cancellation can release it.
			select {
			case <-stopped:
			case <-time.After(2 * time.Second):
				t.Fatal("terminal send did not unblock on cancellation")
			}
			if calls != cancelAfter {
				t.Fatalf("got %d flushed calls", calls)
			}
			if _, ok := <-out; ok {
				t.Fatal("cancelled finalizer emitted an extra chunk")
			}
		})
	}
}
