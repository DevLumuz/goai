//go:build e2e

package azure

import (
	"context"
	"os"
	"testing"
	"time"

	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// TestE2E_AzureRealtime needs AZURE_OPENAI_ENDPOINT (or AZURE_RESOURCE_NAME),
// AZURE_OPENAI_API_KEY, and AZURE_OPENAI_REALTIME_DEPLOYMENT naming a
// gpt-realtime-2.x deployment.
func TestE2E_AzureRealtime(t *testing.T) {
	deployment := os.Getenv("AZURE_OPENAI_REALTIME_DEPLOYMENT")
	if deployment == "" {
		t.Skip("AZURE_OPENAI_REALTIME_DEPLOYMENT not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	session, err := goai.OpenRealtimeSession(ctx, Realtime(deployment),
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
		case provider.RealtimeResponseFinished:
			if audio == 0 {
				t.Fatal("no audio received")
			}
			t.Logf("finished with %d audio bytes", audio)
			return
		}
	}
	t.Fatalf("session ended early: %v", session.Err())
}
