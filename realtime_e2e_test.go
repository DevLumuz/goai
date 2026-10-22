//go:build e2e

package goai_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/azure"
	"github.com/zendev-sh/goai/provider/google"
	"github.com/zendev-sh/goai/provider/openai"
	"github.com/zendev-sh/goai/provider/vertex"
)

// The realtime E2E suite runs the same scenarios against every configured
// backend and skips what a backend does not declare in its capabilities:
//
//	gemini  GEMINI_API_KEY [GEMINI_LIVE_MODEL]
//	vertex  GOOGLE_CLOUD_PROJECT + Application Default Credentials
//	        [GOOGLE_CLOUD_LOCATION] [VERTEX_LIVE_MODEL]
//	openai  OPENAI_API_KEY [OPENAI_REALTIME_MODEL] [OPENAI_TRANSLATION_MODEL]
//	azure   AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY +
//	        AZURE_OPENAI_REALTIME_DEPLOYMENT [AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT]
//	        [AZURE_OPENAI_TRANSLATE_DEPLOYMENT]
//
// Run: go test -tags=e2e -run TestE2E_Realtime -v .

type e2eBackend struct {
	name  string
	model provider.RealtimeModel
	// openaiFamily marks the OpenAI Realtime protocol (OpenAI and Azure).
	openaiFamily bool
	// toolOptions are the ProviderDefinedOptions of test tools.
	toolOptions map[string]any
	// transcription configures input transcription; nil skips that scenario.
	transcription *provider.TranscriptionConfig
	// metadataKey is the ProviderMetadata key of the backend.
	metadataKey string
}

func cmpEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func e2eBackends(t *testing.T) []e2eBackend {
	t.Helper()
	var backends []e2eBackend
	if os.Getenv("GEMINI_API_KEY") != "" || os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY") != "" {
		backends = append(backends, e2eBackend{
			name:          "gemini",
			model:         google.Live(cmpEnv("GEMINI_LIVE_MODEL", "gemini-3.8-live")),
			toolOptions:   google.LiveToolOptions(google.LiveBlocking, ""),
			transcription: &provider.TranscriptionConfig{Enabled: true},
			metadataKey:   "google",
		})
	}
	if os.Getenv("GOOGLE_CLOUD_PROJECT") != "" {
		backends = append(backends, e2eBackend{
			name:          "vertex",
			model:         vertex.Live(cmpEnv("VERTEX_LIVE_MODEL", "gemini-3.8-live")),
			toolOptions:   google.LiveToolOptions(google.LiveBlocking, ""),
			transcription: &provider.TranscriptionConfig{Enabled: true},
			metadataKey:   "google",
		})
	}
	if os.Getenv("OPENAI_API_KEY") != "" {
		backends = append(backends, e2eBackend{
			name:          "openai",
			model:         openai.Realtime(cmpEnv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")),
			openaiFamily:  true,
			transcription: &provider.TranscriptionConfig{Enabled: true},
			metadataKey:   "openai",
		})
	}
	if deployment := os.Getenv("AZURE_OPENAI_REALTIME_DEPLOYMENT"); deployment != "" {
		b := e2eBackend{name: "azure", model: azure.Realtime(deployment), openaiFamily: true, metadataKey: "azure"}
		// Azure transcribes with a model deployed in the same resource.
		if d := os.Getenv("AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT"); d != "" {
			b.transcription = &provider.TranscriptionConfig{Enabled: true, ProviderOptions: map[string]any{"model": d}}
		}
		backends = append(backends, b)
	}
	if len(backends) == 0 {
		t.Skip("no realtime backend configured")
	}
	return backends
}

// forEachBackend runs scenario as a subtest per backend.
func forEachBackend(t *testing.T, scenario func(t *testing.T, b e2eBackend)) {
	for _, b := range e2eBackends(t) {
		t.Run(b.name, func(t *testing.T) { scenario(t, b) })
	}
}

func openE2E(t *testing.T, ctx context.Context, model provider.RealtimeModel, opts ...goai.RealtimeOption) *goai.RealtimeSession {
	t.Helper()
	opts = append([]goai.RealtimeOption{goai.WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true})}, opts...)
	session, err := goai.OpenRealtimeSession(ctx, model, opts...)
	if err != nil {
		if strings.Contains(err.Error(), "exceeded your current quota") {
			t.Skipf("no quota: %v", err)
		}
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = session.Close() })
	return session
}

// reply collects events up to the end of the next completed response and
// returns its final transcript (or final text) and every event seen.
func reply(t *testing.T, session *goai.RealtimeSession) (string, []provider.RealtimeEvent) {
	t.Helper()
	var text string
	var seen []provider.RealtimeEvent
	for ev := range session.Events() {
		seen = append(seen, ev)
		switch ev.Type {
		case provider.RealtimeOutputTranscript, provider.RealtimeTextDelta:
			if ev.IsFinal {
				text = ev.Text
			}
		case provider.RealtimeResponseFinished:
			if !ev.Pending {
				return text, seen
			}
		}
	}
	t.Fatalf("session ended before the response finished: %v", session.Err())
	return "", nil
}

func mustContain(t *testing.T, got, want string) {
	t.Helper()
	if !strings.Contains(strings.ToLower(got), want) {
		t.Fatalf("reply %q does not contain %q", got, want)
	}
}

func TestE2E_RealtimeTextToAudio(t *testing.T) {
	forEachBackend(t, func(t *testing.T, b e2eBackend) {
		ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
		defer cancel()
		session := openE2E(t, ctx, b.model, goai.WithRealtimeSystem("Answer in one short sentence."))
		if err := session.SendText(ctx, "Say hello."); err != nil {
			t.Fatal(err)
		}
		transcript, seen := reply(t, session)
		audio := 0
		for _, ev := range seen {
			if ev.Type == provider.RealtimeOutputAudio {
				audio += len(ev.Audio)
				if ev.AudioFormat.SampleRate == 0 {
					t.Fatalf("audio without a format: %+v", ev.AudioFormat)
				}
			}
		}
		if audio == 0 || transcript == "" {
			t.Fatalf("audio %d bytes, transcript %q", audio, transcript)
		}
	})
}

func TestE2E_RealtimeToolCall(t *testing.T) {
	forEachBackend(t, func(t *testing.T, b e2eBackend) {
		ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
		defer cancel()
		called := make(chan string, 1)
		weather := goai.Tool{
			Name:                   "get_weather",
			Description:            "Returns the current weather for a city.",
			InputSchema:            json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
			ProviderDefinedOptions: b.toolOptions,
			Execute: func(_ context.Context, input json.RawMessage) (string, error) {
				called <- string(input)
				return `{"condition":"snowing","celsius":-3}`, nil
			},
		}
		session := openE2E(t, ctx, b.model, goai.WithRealtimeTools(weather))
		if err := session.SendText(ctx, "What is the weather in Lima right now? Use the tool, then tell me."); err != nil {
			t.Fatal(err)
		}
		ran := false
		for {
			transcript, _ := reply(t, session)
			select {
			case input := <-called:
				t.Logf("tool input %s", input)
				ran = true
			default:
			}
			if ran && strings.Contains(strings.ToLower(transcript), "snow") {
				return
			}
		}
	})
}

// TestE2E_RealtimeDeclaredTool answers a tool without Execute from the
// application, as a UI or a device would.
func TestE2E_RealtimeDeclaredTool(t *testing.T) {
	forEachBackend(t, func(t *testing.T, b e2eBackend) {
		ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
		defer cancel()
		vault := goai.Tool{
			Name:                   "read_vault_password",
			Description:            "Reads the vault password.",
			InputSchema:            json.RawMessage(`{"type":"object","properties":{}}`),
			ProviderDefinedOptions: b.toolOptions,
		}
		session := openE2E(t, ctx, b.model, goai.WithRealtimeTools(vault))
		if err := session.SendText(ctx, "Read the vault password with the tool and say it."); err != nil {
			t.Fatal(err)
		}
		answered := false
		for ev := range session.Events() {
			if ev.Type == provider.RealtimeToolCallEvent && ev.ToolCall.Name == "read_vault_password" {
				if err := session.SendToolResult(ctx, provider.RealtimeToolResult{ToolCallID: ev.ToolCall.ID, Output: json.RawMessage(`{"password":"tangerine"}`)}); err != nil {
					t.Fatal(err)
				}
				answered = true
				break
			}
		}
		if !answered {
			t.Fatalf("the model never called the tool: %v", session.Err())
		}
		for {
			if transcript, _ := reply(t, session); strings.Contains(strings.ToLower(transcript), "tangerine") {
				return
			}
		}
	})
}

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

func TestE2E_RealtimeImageInput(t *testing.T) {
	forEachBackend(t, func(t *testing.T, b e2eBackend) {
		if !provider.RealtimeCapabilitiesOf(b.model).ImageInput {
			t.Skip("no image input")
		}
		ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
		defer cancel()
		session := openE2E(t, ctx, b.model)
		if err := session.InputImage(ctx, provider.RealtimeMedia{MIMEType: "image/png", Data: solidPNG(t, color.RGBA{R: 255, A: 255})}); err != nil {
			t.Fatal(err)
		}
		time.Sleep(2 * time.Second) // realtime frames are not ordered against text
		if err := session.SendText(ctx, "What single color fills the image? Answer with one word."); err != nil {
			t.Fatal(err)
		}
		transcript, _ := reply(t, session)
		mustContain(t, transcript, "red")
	})
}

func TestE2E_RealtimeConversationHistory(t *testing.T) {
	forEachBackend(t, func(t *testing.T, b e2eBackend) {
		ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
		defer cancel()
		session := openE2E(t, ctx, b.model)
		err := session.AppendConversation(ctx, provider.RealtimeConversation{RequestResponse: true, Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "My favorite fruit is the mango."}}},
			{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartText, Text: "Noted, mango it is."}}},
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "What is my favorite fruit? One word."}}},
		}})
		if err != nil {
			t.Fatal(err)
		}
		transcript, _ := reply(t, session)
		mustContain(t, transcript, "mango")
	})
}

func TestE2E_RealtimeSessionUpdate(t *testing.T) {
	forEachBackend(t, func(t *testing.T, b e2eBackend) {
		ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
		defer cancel()
		session := openE2E(t, ctx, b.model)
		system := "Whatever the user says, reply only with the word pineapple."
		err := session.UpdateSession(ctx, goai.RealtimeUpdate{System: &system})
		if !provider.RealtimeCapabilitiesOf(b.model).RuntimeSessionUpdate {
			if !errors.Is(err, provider.ErrRealtimeUnsupported) {
				t.Fatalf("an unsupported update must fail: %v", err)
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		time.Sleep(time.Second)
		if err := session.SendText(ctx, "How are you today?"); err != nil {
			t.Fatal(err)
		}
		transcript, _ := reply(t, session)
		mustContain(t, transcript, "pineapple")
	})
}

func TestE2E_RealtimeRawEvents(t *testing.T) {
	forEachBackend(t, func(t *testing.T, b e2eBackend) {
		ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
		defer cancel()
		session := openE2E(t, ctx, b.model, goai.WithRealtimeRawEvents(true))
		if err := session.SendText(ctx, "Say hi."); err != nil {
			t.Fatal(err)
		}
		_, seen := reply(t, session)
		for _, ev := range seen {
			if ev.Type == provider.RealtimeProviderEvent && json.Valid(ev.Raw) {
				return
			}
		}
		t.Fatal("no raw provider events")
	})
}

// speech asks the backend to say sentence and returns its audio.
func speech(t *testing.T, ctx context.Context, model provider.RealtimeModel, sentence string) ([]byte, provider.AudioFormat) {
	t.Helper()
	session := openE2E(t, ctx, model, goai.WithRealtimeSystem("Repeat exactly what the user asks you to say, nothing else."))
	if err := session.SendText(ctx, "Say: "+sentence); err != nil {
		t.Fatal(err)
	}
	_, seen := reply(t, session)
	var audio []byte
	var format provider.AudioFormat
	for _, ev := range seen {
		if ev.Type == provider.RealtimeOutputAudio {
			audio = append(audio, ev.Audio...)
			format = ev.AudioFormat
		}
	}
	if len(audio) == 0 {
		t.Fatal("no speech generated")
	}
	_ = session.Close()
	return audio, format
}

// resampleTo converts mono PCM16 to rate by linear interpolation.
func resampleTo(in []byte, from, to int) []byte {
	if from == to {
		return in
	}
	n := len(in) / 2
	sample := func(i int) float64 { return float64(int16(binary.LittleEndian.Uint16(in[2*i:]))) }
	step := float64(from) / float64(to)
	var out []byte
	for pos := 0.0; int(pos)+1 < n; pos += step {
		i, frac := int(pos), pos-float64(int(pos))
		out = binary.LittleEndian.AppendUint16(out, uint16(int16(sample(i)*(1-frac)+sample(i+1)*frac)))
	}
	return out
}

// streamAudio sends audio in 100 ms frames at speaking pace, then silence.
// It may run on its own goroutine, so it reports with t.Errorf.
func streamAudio(t *testing.T, ctx context.Context, session *goai.RealtimeSession, audio []byte, rate int) {
	frame := rate / 10 * 2
	silence := make([]byte, frame)
	for i := 0; i < len(audio)+15*frame; i += frame {
		chunk := silence
		if i < len(audio) {
			chunk = audio[i:min(i+frame, len(audio))]
		}
		if err := session.InputAudio(ctx, chunk); err != nil {
			t.Errorf("sending audio: %v", err)
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// TestE2E_RealtimeInputTranscription speaks the model's own voice back to it.
func TestE2E_RealtimeInputTranscription(t *testing.T) {
	forEachBackend(t, func(t *testing.T, b e2eBackend) {
		if b.transcription == nil {
			t.Skip("no transcription model configured")
		}
		ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
		defer cancel()
		audio, format := speech(t, ctx, b.model, "the quick brown fox jumps over the lazy dog")
		in := provider.RealtimeCapabilitiesOf(b.model).SupportedInputAudio[0]
		listener := openE2E(t, ctx, b.model, goai.WithRealtimeInputTranscription(*b.transcription))
		go streamAudio(t, ctx, listener, resampleTo(audio, format.SampleRate, in.SampleRate), in.SampleRate)
		var started bool
		for ev := range listener.Events() {
			switch ev.Type {
			case provider.RealtimeInputSpeechStarted:
				started = true
			case provider.RealtimeInputTranscript:
				if ev.IsFinal {
					mustContain(t, ev.Text, "fox")
					if !started {
						t.Error("no speech start before the transcript")
					}
					return
				}
			}
		}
		t.Fatalf("no final input transcript: %v", listener.Err())
	})
}

func TestE2E_RealtimeInterrupt(t *testing.T) {
	forEachBackend(t, func(t *testing.T, b e2eBackend) {
		if !provider.RealtimeCapabilitiesOf(b.model).OutputAudioTruncation {
			t.Skip("no explicit interruption")
		}
		ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
		defer cancel()
		session := openE2E(t, ctx, b.model)
		if err := session.SendText(ctx, "Count slowly from one to fifty."); err != nil {
			t.Fatal(err)
		}
		received := 0
		interrupted := false
		for ev := range session.Events() {
			switch ev.Type {
			case provider.RealtimeOutputAudio:
				received += len(ev.Audio)
				if !interrupted && received > 48_000 { // about a second of audio
					interrupted = true
					err := session.Interrupt(ctx, &provider.PlaybackCursor{ResponseID: ev.ResponseID, ItemID: ev.ItemID, PlayedFor: 500 * time.Millisecond})
					if err != nil {
						t.Fatal(err)
					}
				}
			case provider.RealtimeError:
				t.Fatalf("error: %v", ev.Error)
			case provider.RealtimeResponseFinished:
				if ev.FinishReason != provider.FinishInterrupted {
					t.Fatalf("finish = %s, want interrupted", ev.FinishReason)
				}
				return
			}
		}
		t.Fatalf("session ended early: %v", session.Err())
	})
}

func TestE2E_RealtimeResumption(t *testing.T) {
	forEachBackend(t, func(t *testing.T, b e2eBackend) {
		if !provider.RealtimeCapabilitiesOf(b.model).SessionResumption {
			t.Skip("no session resumption")
		}
		ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
		defer cancel()
		first := openE2E(t, ctx, b.model, goai.WithRealtimeResumption(""))
		if err := first.SendText(ctx, "Remember this code word: zebra. Just say OK."); err != nil {
			t.Fatal(err)
		}
		var token string
		for ev := range first.Events() {
			if ev.Type == provider.RealtimeSessionResumeState && ev.ResumeToken != "" {
				token = ev.ResumeToken
			}
			if ev.Type == provider.RealtimeResponseFinished && !ev.Pending && token != "" {
				break
			}
		}
		_ = first.Close()
		second := openE2E(t, ctx, b.model, goai.WithRealtimeResumption(token))
		if err := second.SendText(ctx, "What was the code word? One word."); err != nil {
			t.Fatal(err)
		}
		transcript, _ := reply(t, second)
		mustContain(t, transcript, "zebra")
	})
}

// TestE2E_RealtimeOutOfBandResponse classifies the user's words in a response
// kept out of the conversation (OpenAI Realtime protocol).
func TestE2E_RealtimeOutOfBandResponse(t *testing.T) {
	forEachBackend(t, func(t *testing.T, b e2eBackend) {
		if !b.openaiFamily {
			t.Skip("out-of-band responses are an OpenAI Realtime feature")
		}
		ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
		defer cancel()
		session := openE2E(t, ctx, b.model, goai.WithRealtimeTurnDetection(provider.TurnDetection{DisableAutoResponse: true}))
		opts, err := openai.RealtimeResponse(openai.RealtimeResponseOptions{
			OutOfBand:        true,
			OutputModalities: []string{"text"},
			Instructions:     "Reply with exactly one word: banana.",
			Metadata:         map[string]string{"purpose": "e2e"},
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := session.RequestResponseWith(ctx, opts); err != nil {
			t.Fatal(err)
		}
		text, seen := reply(t, session)
		mustContain(t, text, "banana")
		finished := seen[len(seen)-1]
		meta, _ := finished.ProviderMetadata[b.metadataKey].(map[string]any)
		if m, _ := meta["metadata"].(map[string]any); m["purpose"] != "e2e" {
			t.Fatalf("metadata = %+v", finished.ProviderMetadata)
		}
	})
}

// TestE2E_RealtimeResponseQueue sends text while a response is still active:
// the sequencer must not trigger conversation_already_has_active_response.
func TestE2E_RealtimeResponseQueue(t *testing.T) {
	forEachBackend(t, func(t *testing.T, b e2eBackend) {
		if !b.openaiFamily {
			t.Skip("response sequencing is an OpenAI Realtime concern")
		}
		ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
		defer cancel()
		session := openE2E(t, ctx, b.model)
		if err := session.SendText(ctx, "Say the word one."); err != nil {
			t.Fatal(err)
		}
		if err := session.SendText(ctx, "Now say the word two."); err != nil {
			t.Fatal(err)
		}
		finished := 0
		for ev := range session.Events() {
			switch ev.Type {
			case provider.RealtimeError:
				t.Fatalf("error: %v", ev.Error)
			case provider.RealtimeResponseFinished:
				if finished++; finished == 2 {
					return
				}
			}
		}
		t.Fatalf("session ended early: %v", session.Err())
	})
}

// TestE2E_RealtimeTranslation translates speech generated by the realtime
// model of the same backend.
func TestE2E_RealtimeTranslation(t *testing.T) {
	type pair struct {
		name                string
		speaker, translator provider.RealtimeModel
	}
	var pairs []pair
	if os.Getenv("OPENAI_API_KEY") != "" {
		pairs = append(pairs, pair{"openai", openai.Realtime(cmpEnv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")),
			openai.Translation(cmpEnv("OPENAI_TRANSLATION_MODEL", "gpt-realtime-translate"))})
	}
	if rt, tr := os.Getenv("AZURE_OPENAI_REALTIME_DEPLOYMENT"), os.Getenv("AZURE_OPENAI_TRANSLATE_DEPLOYMENT"); rt != "" && tr != "" {
		pairs = append(pairs, pair{"azure", azure.Realtime(rt), azure.Translation(tr)})
	}
	if len(pairs) == 0 {
		t.Skip("no translation backend configured")
	}
	for _, p := range pairs {
		t.Run(p.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
			defer cancel()
			audio, format := speech(t, ctx, p.speaker, "Good morning, the weather is beautiful today.")
			session := openE2E(t, ctx, p.translator,
				goai.WithRealtimeInputTranscription(provider.TranscriptionConfig{Enabled: true}),
				goai.WithRealtimeProviderOptions(openai.TranslationProviderOptions(openai.TranslationConfig{Language: "es"})),
			)
			var mu sync.Mutex
			var translated strings.Builder
			var translatedAudio int
			done := make(chan struct{})
			go func() {
				defer close(done)
				for ev := range session.Events() {
					mu.Lock()
					switch ev.Type {
					case provider.RealtimeOutputTranscript:
						translated.WriteString(ev.Text)
					case provider.RealtimeOutputAudio:
						translatedAudio += len(ev.Audio)
					case provider.RealtimeError:
						t.Errorf("error: %v", ev.Error)
					}
					mu.Unlock()
				}
			}()
			streamAudio(t, ctx, session, resampleTo(audio, format.SampleRate, 24_000), 24_000)
			time.Sleep(3 * time.Second)
			if err := session.Close(); err != nil {
				t.Fatal(err)
			}
			<-done
			mu.Lock()
			defer mu.Unlock()
			t.Logf("translated: %q, %d audio bytes", translated.String(), translatedAudio)
			if translatedAudio == 0 || translated.Len() == 0 {
				t.Fatal("no translation")
			}
		})
	}
}

// TestE2E_RealtimeVertexOnly covers what Gemini Live offers only on Vertex AI:
// transparent resumption and explicit VAD signals.
func TestE2E_RealtimeVertexOnly(t *testing.T) {
	if os.Getenv("GOOGLE_CLOUD_PROJECT") == "" {
		t.Skip("GOOGLE_CLOUD_PROJECT not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	model := vertex.Live(cmpEnv("VERTEX_LIVE_MODEL", "gemini-3.8-live"))
	audio, format := speech(t, ctx, model, "the quick brown fox jumps over the lazy dog")
	session := openE2E(t, ctx, model,
		goai.WithRealtimeRawEvents(true),
		goai.WithRealtimeProviderOptions(google.LiveProviderOptions(google.LiveConfig{
			SessionResumption: &google.LiveSessionResumption{Transparent: true},
			ExplicitVADSignal: true,
		})),
	)
	go streamAudio(t, ctx, session, resampleTo(audio, format.SampleRate, 16_000), 16_000)
	var index, vadSignal bool
	for ev := range session.Events() {
		if ev.Type == provider.RealtimeSessionResumeState {
			meta, _ := ev.ProviderMetadata["google"].(map[string]any)
			_, index = meta["lastConsumedClientMessageIndex"]
		}
		if ev.Type == provider.RealtimeProviderEvent && strings.Contains(string(ev.Raw), "voiceActivityDetectionSignal") {
			vadSignal = true
		}
		if ev.Type == provider.RealtimeResponseFinished && !ev.Pending {
			break
		}
	}
	if !index {
		t.Error("transparent resumption reported no consumed message index")
	}
	if !vadSignal {
		t.Error("no explicit VAD signal")
	}
}
