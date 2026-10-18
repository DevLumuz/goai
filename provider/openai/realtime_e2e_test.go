//go:build e2e

package openai

import (
	"context"
	"encoding/json"
	"os"
	"sync/atomic"
	"testing"
	"time"

	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// realtimeE2EModel needs OPENAI_API_KEY; OPENAI_REALTIME_MODEL overrides the model.
func realtimeE2EModel(t *testing.T) provider.RealtimeModel {
	t.Helper()
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	id := os.Getenv("OPENAI_REALTIME_MODEL")
	if id == "" {
		id = "gpt-realtime-2.1-mini"
	}
	return Realtime(id)
}

func TestE2E_RealtimeTextToAudio(t *testing.T) {
	runRealtimeTextToAudio(t, realtimeE2EModel(t))
}

func TestE2E_RealtimeToolCall(t *testing.T) {
	runRealtimeToolCall(t, realtimeE2EModel(t))
}

func runRealtimeTextToAudio(t *testing.T, model provider.RealtimeModel) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	session, err := goai.OpenRealtimeSession(ctx, model,
		goai.WithRealtimeSystem("Answer in one short sentence."),
		goai.WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true}),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close() }()
	if err := session.SendText(ctx, "Say hello."); err != nil {
		t.Fatal(err)
	}
	audio := 0
	for ev := range session.Events() {
		switch ev.Type {
		case provider.RealtimeOutputAudio:
			audio += len(ev.Audio)
		case provider.RealtimeOutputTranscript:
			if ev.IsFinal {
				t.Logf("transcript: %q", ev.Text)
			}
		case provider.RealtimeError:
			t.Logf("error event: %v", ev.Error)
		case provider.RealtimeResponseFinished:
			t.Logf("finished (%s): %d audio bytes, usage %+v", ev.FinishReason, audio, ev.Usage)
			if audio == 0 {
				t.Fatal("no audio received")
			}
			return
		}
	}
	t.Fatalf("session ended early: %v", session.Err())
}

func runRealtimeToolCall(t *testing.T, model provider.RealtimeModel) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	var executed atomic.Bool
	weather := goai.Tool{
		Name:        "get_weather",
		Description: "Returns the current weather for a city.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			executed.Store(true)
			return `{"condition":"sunny","celsius":21}`, nil
		},
	}
	session, err := goai.OpenRealtimeSession(ctx, model,
		goai.WithRealtimeTools(weather),
		goai.WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true}),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close() }()
	if err := session.SendText(ctx, "What is the weather in Lima right now? Use the tool."); err != nil {
		t.Fatal(err)
	}
	for ev := range session.Events() {
		switch ev.Type {
		case provider.RealtimeToolCallEvent:
			t.Logf("tool call %s %s", ev.ToolCall.Name, ev.ToolCall.Input)
		case provider.RealtimeOutputTranscript:
			if ev.IsFinal {
				t.Logf("transcript: %q", ev.Text)
			}
		case provider.RealtimeResponseFinished:
			if ev.FinishReason != provider.FinishToolCalls && executed.Load() {
				return
			}
		}
	}
	t.Fatalf("session ended early (tool executed: %v): %v", executed.Load(), session.Err())
}
