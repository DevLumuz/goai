package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/zendev-sh/goai/provider"
)

// translationServer is a minimal translation endpoint: it reports
// session.created, sends the test's events once configured (after answering
// the first session.update when configured is set), and answers session.close.
func translationServer(t *testing.T, configured bool, events []string, received chan<- string) (*httptest.Server, <-chan *http.Request) {
	t.Helper()
	requests := make(chan *http.Request, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests <- r.Clone(context.Background())
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		ctx := r.Context()
		liveWrite(ctx, conn, `{"type":"session.created","session":{"id":"sess_tr"}}`)
		sendEvents := func() {
			for _, ev := range events {
				liveWrite(ctx, conn, ev)
			}
			events = nil
		}
		if !configured {
			sendEvents()
		}
		for {
			ev, raw, ok := liveRead(ctx, conn)
			if !ok {
				return
			}
			received <- raw
			switch ev["type"] {
			case "session.update":
				liveWrite(ctx, conn, `{"type":"session.updated"}`)
				sendEvents()
			case "session.close":
				liveWrite(ctx, conn, `{"type":"session.closed"}`)
				_ = conn.Close(websocket.StatusNormalClosure, "")
				return
			}
		}
	}))
	t.Cleanup(server.Close)
	return server, requests
}

func TestTranslationSession(t *testing.T) {
	liveNoLeaks(t)

	received := make(chan string, 16)
	server, requests := translationServer(t, true, []string{
		`{"type":"session.input_transcript.delta","delta":"hola","elapsed_ms":200}`,
		`{"type":"session.output_transcript.delta","delta":"hello","elapsed_ms":400}`,
		`{"type":"session.output_audio.delta","delta":"AQI=","sample_rate":24000,"channels":1,"format":"pcm16","elapsed_ms":400}`,
		`{"type":"session.future_event"}`,
		`{"type":"error","error":{"type":"invalid_request_error","code":"bad","message":"oops"}}`,
	}, received)

	conn, err := Translation("gpt-realtime-translate", WithAPIKey("k"), WithBaseURL(server.URL+"/v1")).OpenRealtime(t.Context(), provider.RealtimeParams{
		InputTranscription:  provider.TranscriptionConfig{Enabled: true},
		OutputTranscription: provider.TranscriptionConfig{Enabled: true},
		ProviderOptions:     TranslationProviderOptions(TranslationConfig{Language: "en", NoiseReduction: "near_field"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	r := <-requests
	if r.URL.Path != "/v1/realtime/translations" || r.URL.Query().Get("model") != "gpt-realtime-translate" || r.Header.Get("Authorization") != "Bearer k" {
		t.Errorf("request = %s %v", r.URL, r.Header)
	}
	var update map[string]any
	_ = json.Unmarshal([]byte(<-received), &update)
	assertJSON(t, update, `{"type":"session.update","session":{"audio":{
		"input":{"transcription":{"model":"gpt-live-transcribe"},"noise_reduction":{"type":"near_field"}},
		"output":{"language":"en"}}}}`)

	if ev := liveNext(t, conn); ev.Type != provider.RealtimeSessionReady || ev.SessionID != "sess_tr" {
		t.Fatalf("ready = %+v", ev)
	}
	elapsed := func(ev provider.RealtimeEvent) any {
		return ev.ProviderMetadata["openai"].(map[string]any)["elapsedMs"]
	}
	if ev := liveNext(t, conn); ev.Type != provider.RealtimeInputTranscript || ev.Text != "hola" || elapsed(ev) != int64(200) {
		t.Errorf("source transcript = %+v", ev)
	}
	if ev := liveNext(t, conn); ev.Type != provider.RealtimeOutputTranscript || ev.Text != "hello" {
		t.Errorf("translated transcript = %+v", ev)
	}
	if ev := liveNext(t, conn); ev.Type != provider.RealtimeOutputAudio || string(ev.Audio) != "\x01\x02" || ev.AudioFormat != translationInputFormat {
		t.Errorf("translated audio = %+v", ev)
	}
	if ev := liveNext(t, conn); ev.Type != provider.RealtimeProviderEvent {
		t.Errorf("unknown event = %+v", ev)
	}
	if ev := liveNext(t, conn); ev.Type != provider.RealtimeError || ev.Error == nil {
		t.Errorf("error = %+v", ev)
	}

	ctx := t.Context()
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioAppend, Audio: []byte{1, 2}}); err != nil {
		t.Fatal(err)
	}
	if got := <-received; got != `{"audio":"AQI=","type":"session.input_audio_buffer.append"}` {
		t.Errorf("audio = %s", got)
	}
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: &provider.RealtimeSessionUpdate{
		ProviderOptions: TranslationProviderOptions(TranslationConfig{Language: "fr"}),
	}}); err != nil {
		t.Fatal(err)
	}
	if got := <-received; got != `{"session":{"audio":{"output":{"language":"fr"}}},"type":"session.update"}` {
		t.Errorf("update = %s", got)
	}
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeRawEvent, Raw: json.RawMessage(`{"type":"x"}`)}); err != nil {
		t.Fatal(err)
	}
	<-received
	system := "x"
	for _, c := range []struct {
		ev   provider.RealtimeClientEvent
		want error
	}{
		{provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioAppend}, provider.ErrInvalidRealtimeConfig},
		{provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: &provider.RealtimeSessionUpdate{}}, provider.ErrInvalidRealtimeConfig},
		{provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: &provider.RealtimeSessionUpdate{System: &system, ProviderOptions: map[string]any{"a": 1}}}, provider.ErrRealtimeUnsupported},
		{provider.RealtimeClientEvent{Type: provider.RealtimeRawEvent, Raw: json.RawMessage(`[]`)}, provider.ErrInvalidRealtimeConfig},
		{provider.RealtimeClientEvent{Type: provider.RealtimeTextInput, Text: "hi"}, provider.ErrRealtimeUnsupported},
	} {
		if err := conn.Send(ctx, c.ev); !errors.Is(err, c.want) {
			t.Errorf("%s: err = %v, want %v", c.ev.Type, err, c.want)
		}
	}

	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
	if got := <-received; got != `{"type":"session.close"}` {
		t.Errorf("close = %s", got)
	}
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioAppend, Audio: []byte{1}}); !errors.Is(err, provider.ErrRealtimeSessionClosed) {
		t.Errorf("send after close = %v", err)
	}
}

func TestTranslationDefaultsAndRawEvents(t *testing.T) {
	liveNoLeaks(t)

	received := make(chan string, 8)
	server, _ := translationServer(t, false, []string{`{"type":"session.output_transcript.delta","delta":"hidden"}`, `{"type":"session.input_transcript.delta","delta":"hidden"}`}, received)
	conn, err := Translation("m", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{IncludeRawEvents: true})
	if err != nil {
		t.Fatal(err)
	}
	// Without settings, no session.update is sent and transcripts stay hidden.
	for i, want := range []provider.RealtimeEventType{
		provider.RealtimeProviderEvent, provider.RealtimeSessionReady, provider.RealtimeProviderEvent, provider.RealtimeProviderEvent,
	} {
		if ev := liveNext(t, conn); ev.Type != want {
			t.Fatalf("event %d = %+v, want %s", i, ev, want)
		}
	}
	select {
	case got := <-received:
		t.Fatalf("unexpected client message %s", got)
	case <-time.After(50 * time.Millisecond):
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestTranslationValidation(t *testing.T) {
	model := Translation("m", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1"))
	for _, c := range []struct {
		params provider.RealtimeParams
		want   error
	}{
		{provider.RealtimeParams{System: "x"}, provider.ErrRealtimeUnsupported},
		{provider.RealtimeParams{TurnDetection: provider.TurnDetection{Mode: provider.TurnModeManual}}, provider.ErrRealtimeUnsupported},
		{provider.RealtimeParams{InputAudio: livePCMU}, provider.ErrUnsupportedAudioFormat},
		{provider.RealtimeParams{OutputAudio: livePCM24k}, provider.ErrUnsupportedAudioFormat},
		{provider.RealtimeParams{OutputTranscription: provider.TranscriptionConfig{Enabled: true, Language: "es"}}, provider.ErrRealtimeUnsupported},
		{provider.RealtimeParams{MaxOutputTokens: -1}, provider.ErrInvalidRealtimeConfig},
	} {
		if _, err := model.OpenRealtime(t.Context(), c.params); !errors.Is(err, c.want) {
			t.Errorf("%+v: err = %v, want %v", c.params, err, c.want)
		}
	}
	session, err := buildTranslationSession(provider.RealtimeParams{InputTranscription: provider.TranscriptionConfig{Enabled: true, ProviderOptions: map[string]any{"model": "custom"}}})
	if err != nil {
		t.Fatal(err)
	}
	assertJSON(t, session, `{"audio":{"input":{"transcription":{"model":"custom"}}}}`)
	assertJSON(t, TranslationProviderOptions(TranslationConfig{}), `{}`)
	if caps := provider.RealtimeCapabilitiesOf(model); !caps.InputAudio || caps.TextInput || len(caps.SupportedInputAudio) != 1 || model.ModelID() != "m" {
		t.Errorf("capabilities = %+v", caps)
	}
	t.Setenv("OPENAI_API_KEY", "")
	if _, err := Translation("m", WithBaseURL("http://127.0.0.1:1")).OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
		t.Error("missing credentials must fail")
	}
}

func TestTranslationOpenErrors(t *testing.T) {
	liveNoLeaks(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		liveWrite(r.Context(), conn, `{"type":"error","error":{"code":"model_not_found","message":"no such model"}}`)
		_, _, _ = conn.Read(r.Context())
	}))
	defer server.Close()
	if _, err := Translation("m", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected the setup error")
	}
}

func FuzzTranslationTranslate(f *testing.F) {
	for _, seed := range []string{
		`{"type":"session.output_audio.delta","delta":"AQI=","sample_rate":24000,"channels":1}`,
		`{"type":"session.output_audio.delta","delta":"!!"}`,
		`{"type":"session.input_transcript.delta","delta":"x","elapsed_ms":-1}`,
		`{"type":"error"}`,
		`{"type":"session.closed"}`,
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(_ *testing.T, data []byte) {
		var ev translationEvent
		if json.Unmarshal(data, &ev) != nil {
			return
		}
		c := &translationConn{inputTr: true, outputTr: true}
		_ = c.translateMessage(ev, data)
	})
}

func TestTranslationConnectionLost(t *testing.T) {
	liveNoLeaks(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		liveWrite(r.Context(), conn, `{"type":"session.created","session":{"id":"s"}}`)
		_ = conn.Close(websocket.StatusInternalError, "server failure")
	}))
	defer server.Close()

	conn, err := Translation("m", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range conn.Events() {
	}
	if err := conn.Err(); err == nil || !strings.Contains(err.Error(), "server failure") {
		t.Fatalf("Err = %v", err)
	}
	_ = conn.Close()
}
