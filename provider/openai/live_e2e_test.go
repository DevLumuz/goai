//go:build e2e

package openai

import (
	"context"
	"os"
	"testing"
	"time"

	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// TestE2E_LiveResponsesDelegation needs OPENAI_API_KEY. OPENAI_LIVE_MODEL and
// OPENAI_LIVE_DELEGATE_MODEL override the voice and backend models.
func TestE2E_LiveResponsesDelegation(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	model := os.Getenv("OPENAI_LIVE_MODEL")
	if model == "" {
		model = "gpt-live-1"
	}
	backend := os.Getenv("OPENAI_LIVE_DELEGATE_MODEL")
	if backend == "" {
		backend = "gpt-6-luna"
	}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	session, err := goai.OpenRealtimeSession(ctx, Live(model),
		goai.WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true}),
		goai.WithRealtimeProviderOptions(LiveResponsesDelegation(backend)),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close() }()
	if err := session.SendText(ctx, "Say hello in one short sentence."); err != nil {
		t.Fatal(err)
	}

	audio := 0
	idle := time.NewTimer(30 * time.Second)
	defer idle.Stop()
	for {
		select {
		case ev, ok := <-session.Events():
			if !ok {
				t.Fatalf("session ended early: %v", session.Err())
			}
			switch ev.Type {
			case provider.RealtimeOutputAudio:
				audio += len(ev.Audio)
				idle.Reset(3 * time.Second)
			case provider.RealtimeOutputTranscript:
				t.Logf("transcript delta: %q", ev.Text)
			case provider.RealtimeError:
				t.Logf("error event: %v", ev.Error)
			}
		case <-idle.C:
			if audio == 0 {
				t.Fatal("no audio received")
			}
			t.Logf("received %d audio bytes", audio)
			return
		}
	}
}
