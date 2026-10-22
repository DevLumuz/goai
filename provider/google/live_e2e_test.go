//go:build e2e

package google

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"strings"
	"testing"
	"time"

	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// liveE2EModel returns the Live model under test, skipping when no key is set.
// GEMINI_LIVE_MODEL overrides the default model ID.
func liveE2EModel(t *testing.T) provider.RealtimeModel {
	t.Helper()
	if os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set")
	}
	id := os.Getenv("GEMINI_LIVE_MODEL")
	if id == "" {
		id = "gemini-3.8-live"
	}
	return Live(id)
}

// liveE2ESession opens a session on the model under test and closes it at the end.
func liveE2ESession(t *testing.T, ctx context.Context, model provider.RealtimeModel, opts ...goai.RealtimeOption) *goai.RealtimeSession {
	t.Helper()
	opts = append([]goai.RealtimeOption{goai.WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true})}, opts...)
	session, err := goai.OpenRealtimeSession(ctx, model, opts...)
	if err != nil {
		if strings.Contains(err.Error(), "exceeded your current quota") {
			t.Skipf("no quota for this feature: %v", err)
		}
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = session.Close() })
	return session
}

// replyTranscript returns the final output transcript of the next response.
func replyTranscript(t *testing.T, session *goai.RealtimeSession) (string, []provider.RealtimeEvent) {
	t.Helper()
	var transcript string
	var seen []provider.RealtimeEvent
	for ev := range session.Events() {
		seen = append(seen, ev)
		switch ev.Type {
		case provider.RealtimeOutputTranscript:
			if ev.IsFinal {
				transcript = ev.Text
			}
		case provider.RealtimeResponseFinished:
			if !ev.Pending {
				return transcript, seen
			}
		}
	}
	t.Fatalf("session ended before the response finished: %v", session.Err())
	return "", nil
}

// solidPNG returns a 64x64 PNG filled with c.
func solidPNG(t *testing.T, c color.Color) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	draw.Draw(img, img.Bounds(), &image.Uniform{C: c}, image.Point{}, draw.Src)
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// TestE2E_LiveImageInHistory sends a still image through the ordered
// conversation history, the path for an image a question refers to.
func TestE2E_LiveImageInHistory(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	session := liveE2ESession(t, ctx, liveE2EModel(t))

	dataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(solidPNG(t, color.RGBA{B: 255, A: 255}))
	err := session.AppendConversation(ctx, provider.RealtimeConversation{RequestResponse: true, Messages: []provider.Message{{
		Role: provider.RoleUser,
		Content: []provider.Part{
			{Type: provider.PartImage, URL: dataURL},
			{Type: provider.PartText, Text: "What single color fills this image? Answer with one word."},
		},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	if transcript, _ := replyTranscript(t, session); !strings.Contains(strings.ToLower(transcript), "blue") {
		t.Fatalf("transcript = %q, want the color blue", transcript)
	}
}

// TestE2E_LiveAudioStreamEnd speaks the model's own reply back to it, so the
// test needs no recorded audio. audioStreamEnd closes the stream without ending
// the turn; new audio reopens it, as the API reference describes.
func TestE2E_LiveAudioStreamEnd(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	model := liveE2EModel(t)

	speaker := liveE2ESession(t, ctx, model, goai.WithRealtimeSystem("Repeat exactly what the user asks you to say, nothing else."))
	if err := speaker.SendText(ctx, "Say: the quick brown fox jumps over the lazy dog."); err != nil {
		t.Fatal(err)
	}
	var speech []byte
	for ev := range speaker.Events() {
		if ev.Type == provider.RealtimeOutputAudio {
			speech = append(speech, ev.Audio...)
		}
		if ev.Type == provider.RealtimeResponseFinished && !ev.Pending {
			break
		}
	}
	if len(speech) == 0 {
		t.Fatal("no speech generated")
	}

	listener := liveE2ESession(t, ctx, model, goai.WithRealtimeInputTranscription(provider.TranscriptionConfig{Enabled: true}))
	input := resample24to16(speech)
	for i := 0; i < len(input); i += 3200 { // 100 ms frames at speaking pace
		if err := listener.InputAudio(ctx, input[i:min(i+3200, len(input))]); err != nil {
			t.Fatal(err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err := listener.EndInputAudio(ctx); err != nil {
		t.Fatal(err)
	}
	silence := make([]byte, 3200)
	for range 15 {
		if err := listener.InputAudio(ctx, silence); err != nil {
			t.Fatal(err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	for ev := range listener.Events() {
		if ev.Type == provider.RealtimeInputTranscript && ev.IsFinal {
			if !strings.Contains(strings.ToLower(ev.Text), "fox") {
				t.Fatalf("input transcript = %q", ev.Text)
			}
			return
		}
	}
	t.Fatalf("no final input transcript: %v", listener.Err())
}

// resample24to16 converts 24 kHz mono PCM16 to 16 kHz by linear interpolation.
func resample24to16(in []byte) []byte {
	n := len(in) / 2
	sample := func(i int) float64 { return float64(int16(binary.LittleEndian.Uint16(in[2*i:]))) }
	out := make([]byte, 0, n*2/3*2+2)
	for pos := 0.0; int(pos)+1 < n; pos += 1.5 {
		i, frac := int(pos), pos-float64(int(pos))
		v := int16(sample(i)*(1-frac) + sample(i+1)*frac)
		out = binary.LittleEndian.AppendUint16(out, uint16(v))
	}
	return out
}

// TestE2E_LivePendingInteraction checks interactionStatus on a model that
// keeps working after its first reply.
func TestE2E_LivePendingInteraction(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	if os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set")
	}
	model := Live(cmpOrEnv("GEMINI_LIVE_THINKING_MODEL", "gemini-3.8-live-extended-thinking"))
	session := liveE2ESession(t, ctx, model,
		goai.WithRealtimeRawEvents(true),
		goai.WithRealtimeProviderOptions(LiveProviderOptions(LiveConfig{Thinking: &LiveThinking{Level: "LOW"}})),
	)
	if err := session.SendText(ctx, "What is 17 times 23? Answer in one sentence."); err != nil {
		t.Fatal(err)
	}
	_, seen := replyTranscript(t, session)
	var pending, raw bool
	for _, ev := range seen {
		pending = pending || ev.Type == provider.RealtimeResponseFinished && ev.Pending
		raw = raw || ev.Type == provider.RealtimeProviderEvent
	}
	t.Logf("pending response seen: %v", pending)
	if !raw {
		t.Fatal("WithRealtimeRawEvents delivered no raw messages")
	}
}

func TestE2E_LiveGoogleSearchSources(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	search := goai.Tool{Name: "google_search", ProviderDefinedType: "google.google_search"}
	session := liveE2ESession(t, ctx, liveE2EModel(t), goai.WithRealtimeTools(search))
	if err := session.SendText(ctx, "Search the web: what is the latest stable Go release? Answer briefly."); err != nil {
		t.Fatal(err)
	}
	_, seen := replyTranscript(t, session)
	for _, ev := range seen {
		if ev.Type == provider.RealtimeSources && len(ev.Sources) > 0 {
			return
		}
	}
	t.Fatal("no grounding sources")
}

func cmpOrEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
