//go:build e2e

package openai

import (
	"context"
	"os"
	"strings"
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

// liveTranscript gathers the spoken transcript until the voice stays quiet.
func liveTranscript(t *testing.T, session *goai.RealtimeSession) (string, string) {
	t.Helper()
	var text strings.Builder
	var sessionID string
	idle := time.NewTimer(30 * time.Second)
	defer idle.Stop()
	for {
		select {
		case ev, ok := <-session.Events():
			if !ok {
				t.Fatalf("session ended early: %v", session.Err())
			}
			switch ev.Type {
			case provider.RealtimeSessionReady:
				sessionID = ev.SessionID
			case provider.RealtimeOutputTranscript:
				text.WriteString(ev.Text)
				idle.Reset(3 * time.Second)
			case provider.RealtimeOutputAudio:
				idle.Reset(3 * time.Second)
			}
		case <-idle.C:
			return text.String(), sessionID
		}
	}
}

// TestE2E_LiveFork stores a session and continues it in a fork.
func TestE2E_LiveFork(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	opts := LiveResponsesDelegation(cmpOr(os.Getenv("OPENAI_LIVE_DELEGATE_MODEL"), "gpt-6-luna"))
	opts["store"] = true
	first, err := goai.OpenRealtimeSession(ctx, Live(cmpOr(os.Getenv("OPENAI_LIVE_MODEL"), "gpt-live-1")),
		goai.WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true}),
		goai.WithRealtimeProviderOptions(opts),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := first.SendText(ctx, "Remember this code word: zebra. Just say OK."); err != nil {
		t.Fatal(err)
	}
	_, sourceID := liveTranscript(t, first)
	_ = first.Close()
	if sourceID == "" {
		t.Fatal("no session ID to fork")
	}

	fork, err := goai.OpenRealtimeSession(ctx, LiveFork(sourceID),
		goai.WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true}),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = fork.Close() }()
	if err := fork.SendText(ctx, "What was the code word? One word."); err != nil {
		t.Fatal(err)
	}
	transcript, forkID := liveTranscript(t, fork)
	if forkID == "" || forkID == sourceID {
		t.Fatalf("fork session ID = %q (source %q)", forkID, sourceID)
	}
	if !strings.Contains(strings.ToLower(transcript), "zebra") {
		t.Fatalf("the fork forgot the conversation: %q", transcript)
	}
}

func cmpOr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
