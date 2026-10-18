package openairealtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
	"go.uber.org/goleak"
)

func newTestConn() *conn {
	return &conn{provider: "openai", pending: map[string]bool{}, audioMs: map[string]int64{}, outFormat: pcm24k}
}

func verifyNoLeaks(t *testing.T) {
	t.Helper()
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })
}

// testServer is a minimal Realtime endpoint; handle owns the server socket.
func testServer(t *testing.T, handle func(ctx context.Context, conn *websocket.Conn, r *http.Request)) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		handle(r.Context(), conn, r)
	}))
	t.Cleanup(server.Close)
	return server
}

func readEvent(t *testing.T, ctx context.Context, conn *websocket.Conn) map[string]any {
	t.Helper()
	_, data, err := conn.Read(ctx)
	if err != nil {
		t.Errorf("server read: %v", err)
		return nil
	}
	var ev map[string]any
	if err := json.Unmarshal(data, &ev); err != nil {
		t.Errorf("client sent invalid JSON %s", data)
	}
	return ev
}

func sendEvent(t *testing.T, ctx context.Context, conn *websocket.Conn, ev string) {
	t.Helper()
	if err := conn.Write(ctx, websocket.MessageText, []byte(ev)); err != nil {
		t.Errorf("server write: %v", err)
	}
}

// acceptSession performs the GA handshake and returns the session object.
func acceptSession(t *testing.T, ctx context.Context, conn *websocket.Conn) map[string]any {
	t.Helper()
	sendEvent(t, ctx, conn, `{"type":"session.created","session":{"id":"sess_1"}}`)
	ev := readEvent(t, ctx, conn)
	if ev["type"] != "session.update" {
		t.Errorf("first client event = %v, want session.update", ev["type"])
	}
	sendEvent(t, ctx, conn, `{"type":"session.updated","session":{"id":"sess_1"}}`)
	session, _ := ev["session"].(map[string]any)
	return session
}

func drain(ctx context.Context, conn *websocket.Conn) {
	for {
		if _, _, err := conn.Read(ctx); err != nil {
			return
		}
	}
}

func testModel(baseURL string) provider.RealtimeModel {
	return New(Config{
		Provider: "openai",
		ModelID:  "gpt-realtime-2.1",
		BaseURL:  baseURL + "/v1",
		Headers:  map[string]string{"X-Custom": "yes", "Authorization": "spoofed"},
		Authorize: func(_ context.Context, h http.Header) error {
			h.Set("Authorization", "Bearer sk-test")
			return nil
		},
	})
}

func next(t *testing.T, conn provider.RealtimeConnection) provider.RealtimeEvent {
	t.Helper()
	select {
	case ev, ok := <-conn.Events():
		if !ok {
			t.Fatalf("events closed early; Err() = %v", conn.Err())
		}
		return ev
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for an event")
		return provider.RealtimeEvent{}
	}
}

func TestSessionUpdate(t *testing.T) {
	verifyNoLeaks(t)

	type result struct {
		path, query, auth, custom string
		session                   map[string]any
	}
	got := make(chan result, 1)
	server := testServer(t, func(ctx context.Context, conn *websocket.Conn, r *http.Request) {
		session := acceptSession(t, ctx, conn)
		got <- result{r.URL.Path, r.URL.RawQuery, r.Header.Get("Authorization"), r.Header.Get("X-Custom"), session}
		drain(ctx, conn)
	})

	conn, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{
		System:     "Be brief.",
		Voice:      "marin",
		ToolChoice: "lookup",
		Tools: []provider.ToolDefinition{
			{Name: "lookup", Description: "Looks up", InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)},
			{Name: "noargs"},
		},
		InputAudio:          pcmu8k,
		InputTranscription:  provider.TranscriptionConfig{Enabled: true, Language: "es", ProviderOptions: map[string]any{"keywords": []string{"goai"}}},
		OutputTranscription: provider.TranscriptionConfig{Enabled: true},
		TurnDetection:       provider.TurnDetection{DisableInterruption: true, ProviderOptions: map[string]any{"type": "semantic_vad", "eagerness": "low"}},
		MaxOutputTokens:     512,
		ProviderOptions:     map[string]any{"reasoning": map[string]any{"effort": "low"}, "audio": map[string]any{"output": map[string]any{"speed": 1.1}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()
	if ev := next(t, conn); ev.Type != provider.RealtimeSessionReady {
		t.Fatalf("first event = %s", ev.Type)
	}

	r := <-got
	if r.path != "/v1/realtime" || r.query != "model=gpt-realtime-2.1" {
		t.Errorf("endpoint = %s?%s", r.path, r.query)
	}
	if r.auth != "Bearer sk-test" || r.custom != "yes" {
		t.Errorf("headers: auth=%q custom=%q", r.auth, r.custom)
	}
	body, _ := json.Marshal(r.session)
	for _, want := range []string{
		`"type":"realtime"`,
		`"instructions":"Be brief."`,
		`"output_modalities":["audio"]`,
		`"input":{"format":{"type":"audio/pcmu"}`,
		`"transcription":{"keywords":["goai"],"language":"es","model":"gpt-live-transcribe"}`,
		`"turn_detection":{"eagerness":"low","interrupt_response":false,"type":"semantic_vad"}`,
		`"reasoning":{"effort":"low"}`,
		`"max_output_tokens":512`,
		`"tool_choice":{"name":"lookup","type":"function"}`,
		`"parameters":{"properties":{},"type":"object"}`,
		`"output":{"format":{"rate":24000,"type":"audio/pcm"},"speed":1.1,"voice":"marin"}`,
	} {
		if !strings.Contains(string(body), want) {
			t.Errorf("session missing %s\n%s", want, body)
		}
	}
}

func TestTurnDetectionModes(t *testing.T) {
	manual, _ := buildTurnDetection(provider.TurnDetection{Mode: provider.TurnModeManual})
	if !manual.set || manual.value != nil {
		t.Fatalf("manual turns must send turn_detection: null, got %+v", manual)
	}
	if def, _ := buildTurnDetection(provider.TurnDetection{}); def.set {
		t.Fatal("the default must leave turn_detection to the server")
	}
	noAuto, _ := buildTurnDetection(provider.TurnDetection{DisableAutoResponse: true})
	if v := noAuto.value.(map[string]any); v["type"] != "server_vad" || v["create_response"] != false {
		t.Fatalf("DisableAutoResponse = %+v", v)
	}
	if _, err := buildTurnDetection(provider.TurnDetection{Mode: provider.TurnModeClientVAD}); !errors.Is(err, provider.ErrRealtimeUnsupported) {
		t.Fatalf("client VAD err = %v", err)
	}
}

func TestOpenValidation(t *testing.T) {
	model := testModel("http://127.0.0.1:1")
	pcm16k := provider.AudioFormat{Encoding: provider.AudioPCM16, SampleRate: 16_000, Channels: 1}
	tests := []struct {
		name   string
		params provider.RealtimeParams
		want   error
	}{
		{"input rate", provider.RealtimeParams{InputAudio: pcm16k}, provider.ErrUnsupportedAudioFormat},
		{"output rate", provider.RealtimeParams{OutputAudio: pcm16k}, provider.ErrUnsupportedAudioFormat},
		{"client vad", provider.RealtimeParams{TurnDetection: provider.TurnDetection{Mode: provider.TurnModeClientVAD}}, provider.ErrRealtimeUnsupported},
		{"provider tool", provider.RealtimeParams{Tools: []provider.ToolDefinition{{Name: "web", ProviderDefinedType: "web_search"}}}, provider.ErrRealtimeUnsupported},
		{"bad schema", provider.RealtimeParams{Tools: []provider.ToolDefinition{{Name: "t", InputSchema: json.RawMessage(`{bad`)}}}, provider.ErrInvalidRealtimeConfig},
		{"override type", provider.RealtimeParams{ProviderOptions: map[string]any{"type": "transcription"}}, provider.ErrInvalidRealtimeConfig},
		{"common validation", provider.RealtimeParams{MaxOutputTokens: -1}, provider.ErrInvalidRealtimeConfig},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := model.OpenRealtime(t.Context(), tt.params); !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
		})
	}

	noAuth := New(Config{Provider: "openai", ModelID: "m", BaseURL: "http://127.0.0.1:1"})
	if _, err := noAuth.OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected missing credentials error")
	}
	failingAuth := New(Config{Provider: "openai", ModelID: "m", BaseURL: "http://127.0.0.1:1",
		Authorize: func(context.Context, http.Header) error { return errors.New("no token") }})
	if _, err := failingAuth.OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil || !strings.Contains(err.Error(), "no token") {
		t.Fatalf("auth error = %v", err)
	}
	badScheme := New(Config{Provider: "openai", ModelID: "m", BaseURL: "ftp://x",
		Authorize: func(context.Context, http.Header) error { return nil }})
	if _, err := badScheme.OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected unsupported scheme error")
	}
}

func TestServerEvents(t *testing.T) {
	verifyNoLeaks(t)

	server := testServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		acceptSession(t, ctx, conn)
		for _, ev := range []string{
			`{"type":"input_audio_buffer.speech_started","item_id":"item_u","audio_start_ms":10}`,
			`{"type":"input_audio_buffer.speech_stopped","item_id":"item_u","audio_end_ms":900}`,
			`{"type":"conversation.item.input_audio_transcription.delta","item_id":"item_u","delta":"hola"}`,
			`{"type":"conversation.item.input_audio_transcription.completed","item_id":"item_u","transcript":"hola mundo"}`,
			`{"type":"response.created","response":{"id":"resp_1"}}`,
			`{"type":"response.output_audio.delta","response_id":"resp_1","item_id":"item_a","delta":"AQID"}`,
			`{"type":"response.output_audio_transcript.delta","response_id":"resp_1","item_id":"item_a","delta":"Hi"}`,
			`{"type":"response.output_audio_transcript.done","response_id":"resp_1","item_id":"item_a","transcript":"Hi there"}`,
			`{"type":"response.output_text.delta","response_id":"resp_1","item_id":"item_t","delta":"txt"}`,
			`{"type":"response.done","response":{"id":"resp_1","status":"completed","output":[{"type":"message"}],"usage":{"total_tokens":30,"input_tokens":20,"output_tokens":10,"input_token_details":{"cached_tokens":5}}}}`,
			`{"type":"rate_limits.updated","rate_limits":[{"name":"tokens","limit":1000,"remaining":900,"reset_seconds":1.5}]}`,
			`{"type":"error","error":{"type":"invalid_request_error","code":"response_cancel_not_active","message":"no active response"}}`,
			`{"type":"conversation.item.input_audio_transcription.failed","item_id":"item_u","error":{"type":"server_error","message":"stt failed"}}`,
			`{"type":"conversation.item.added"}`,
		} {
			sendEvent(t, ctx, conn, ev)
		}
		drain(ctx, conn)
	})

	conn, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{
		OutputTranscription: provider.TranscriptionConfig{Enabled: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()

	want := []provider.RealtimeEventType{
		provider.RealtimeSessionReady,
		provider.RealtimeInputSpeechStarted,
		provider.RealtimeInputSpeechStopped,
		provider.RealtimeInputTranscript,
		provider.RealtimeInputTranscript,
		provider.RealtimeResponseStarted,
		provider.RealtimeOutputAudio,
		provider.RealtimeOutputTranscript,
		provider.RealtimeOutputTranscript,
		provider.RealtimeTextDelta,
		provider.RealtimeResponseFinished,
		provider.RealtimeUsageUpdated,
		provider.RealtimeRateLimitsUpdated,
		provider.RealtimeError,
		provider.RealtimeError,
	}
	events := make([]provider.RealtimeEvent, len(want))
	for i := range want {
		events[i] = next(t, conn)
		if events[i].Type != want[i] {
			t.Fatalf("event %d = %s, want %s", i, events[i].Type, want[i])
		}
	}
	if ev := events[4]; !ev.IsFinal || ev.Text != "hola mundo" || ev.ItemID != "item_u" {
		t.Errorf("final input transcript = %+v", ev)
	}
	if ev := events[6]; string(ev.Audio) != "\x01\x02\x03" || ev.ItemID != "item_a" || ev.ResponseID != "resp_1" || ev.AudioFormat != pcm24k {
		t.Errorf("audio = %+v", ev)
	}
	if ev := events[8]; !ev.IsFinal || ev.Text != "Hi there" {
		t.Errorf("final output transcript = %+v", ev)
	}
	wantUsage := provider.Usage{InputTokens: 15, CacheReadTokens: 5, OutputTokens: 10, TotalTokens: 30}
	if ev := events[10]; ev.FinishReason != provider.FinishStop || ev.Usage != wantUsage || ev.ResponseID != "resp_1" {
		t.Errorf("response finished = %+v", ev)
	}
	if rl := events[12].RateLimits; len(rl) != 1 || rl[0].Remaining != 900 || rl[0].ResetAfter != 1500*time.Millisecond {
		t.Errorf("rate limits = %+v", rl)
	}
	if err := events[13].Error; err == nil || !strings.Contains(err.Error(), "response_cancel_not_active") {
		t.Errorf("error event = %v", err)
	}
	if conn.Err() != nil {
		t.Errorf("non-terminal errors must not end the session: %v", conn.Err())
	}
}

func TestOutputTranscriptsOnlyWhenEnabled(t *testing.T) {
	c := newTestConn()
	if evs := c.translate(serverEvent{Type: "response.output_audio_transcript.delta", Delta: "x"}); len(evs) != 0 {
		t.Fatalf("transcript emitted without OutputTranscription: %+v", evs)
	}
}

// scriptedServer answers each response.create with response.created and
// response.done, like the real API, and lets the test push extra events.
func scriptedServer(t *testing.T) (*httptest.Server, chan string, chan string) {
	t.Helper()
	received := make(chan string, 32)
	push := make(chan string, 8)
	server := testServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		acceptSession(t, ctx, conn)
		go func() {
			for {
				select {
				case ev := <-push:
					_ = conn.Write(ctx, websocket.MessageText, []byte(ev))
				case <-ctx.Done():
					return
				}
			}
		}()
		n := 0
		for {
			_, data, err := conn.Read(ctx)
			if err != nil {
				return
			}
			received <- string(data)
			if strings.Contains(string(data), `"type":"response.create"`) {
				n++
				id := fmt.Sprintf("auto_%d", n)
				_ = conn.Write(ctx, websocket.MessageText, []byte(`{"type":"response.created","response":{"id":"`+id+`"}}`))
				_ = conn.Write(ctx, websocket.MessageText, []byte(`{"type":"response.done","response":{"id":"`+id+`","status":"completed"}}`))
			}
		}
	})
	return server, received, push
}

func expectMessages(t *testing.T, received chan string, want ...string) {
	t.Helper()
	for _, w := range want {
		select {
		case got := <-received:
			if got != w {
				t.Errorf("client sent %s\nwant        %s", got, w)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("server did not receive %s", w)
		}
	}
}

func expectNoMessage(t *testing.T, received chan string) {
	t.Helper()
	select {
	case extra := <-received:
		t.Fatalf("unexpected client event: %s", extra)
	case <-time.After(100 * time.Millisecond):
	}
}

// waitFor drains events until one of type typ arrives.
func waitFor(t *testing.T, conn provider.RealtimeConnection, typ provider.RealtimeEventType) provider.RealtimeEvent {
	t.Helper()
	for {
		if ev := next(t, conn); ev.Type == typ {
			return ev
		}
	}
}

func TestClientEvents(t *testing.T) {
	verifyNoLeaks(t)

	server, received, _ := scriptedServer(t)
	conn, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()
	next(t, conn) // ready
	ctx := t.Context()
	send := func(ev provider.RealtimeClientEvent) {
		t.Helper()
		if err := conn.Send(ctx, ev); err != nil {
			t.Fatalf("Send(%s): %v", ev.Type, err)
		}
	}

	send(provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioAppend, Audio: []byte{1, 2, 3}})
	expectMessages(t, received, `{"audio":"AQID","type":"input_audio_buffer.append"}`)
	send(provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioCommit})
	expectMessages(t, received, `{"type":"input_audio_buffer.commit"}`)
	send(provider.RealtimeClientEvent{Type: provider.RealtimeResponseCreate})
	expectMessages(t, received, `{"type":"response.create"}`)
	waitFor(t, conn, provider.RealtimeResponseFinished)
	send(provider.RealtimeClientEvent{Type: provider.RealtimeTextInput, Text: "hola"})
	expectMessages(t, received,
		`{"item":{"content":[{"text":"hola","type":"input_text"}],"role":"user","type":"message"},"type":"conversation.item.create"}`,
		`{"type":"response.create"}`)
	waitFor(t, conn, provider.RealtimeResponseFinished)

	// Without an active response or known audio, an interrupt sends nothing.
	send(provider.RealtimeClientEvent{Type: provider.RealtimeResponseCancel, Cursor: &provider.PlaybackCursor{ItemID: "unknown", PlayedFor: time.Second}})
	expectNoMessage(t, received)

	for _, ev := range []provider.RealtimeClientEvent{
		{Type: provider.RealtimeInputActivityStart},
		{Type: provider.RealtimeInputActivityEnd},
	} {
		if err := conn.Send(ctx, ev); !errors.Is(err, provider.ErrRealtimeUnsupported) {
			t.Errorf("Send(%s) = %v", ev.Type, err)
		}
	}
	for _, ev := range []provider.RealtimeClientEvent{
		{Type: provider.RealtimeInputAudioAppend},
		{Type: provider.RealtimeToolResultEvent},
	} {
		if err := conn.Send(ctx, ev); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
			t.Errorf("Send(%s) = %v", ev.Type, err)
		}
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeResponseCreate}); !errors.Is(err, provider.ErrRealtimeSessionClosed) {
		t.Fatalf("Send after Close = %v", err)
	}
}

// TestResponseSequencing checks that response.create waits for the active
// response and that requests made meanwhile coalesce into one.
func TestResponseSequencing(t *testing.T) {
	verifyNoLeaks(t)

	server, received, push := scriptedServer(t)
	conn, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()
	next(t, conn) // ready
	ctx := t.Context()

	push <- `{"type":"response.created","response":{"id":"server_1"}}`
	waitFor(t, conn, provider.RealtimeResponseStarted)

	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeTextInput, Text: "one"}); err != nil {
		t.Fatal(err)
	}
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeResponseCreate}); err != nil {
		t.Fatal(err)
	}
	expectMessages(t, received, `{"item":{"content":[{"text":"one","type":"input_text"}],"role":"user","type":"message"},"type":"conversation.item.create"}`)
	expectNoMessage(t, received)

	push <- `{"type":"response.done","response":{"id":"server_1","status":"completed"}}`
	expectMessages(t, received, `{"type":"response.create"}`)
	waitFor(t, conn, provider.RealtimeResponseStarted) // auto_1 created by the scripted server
	expectNoMessage(t, received)

	// A race with a server-created response is retried after that response.
	push <- `{"type":"response.created","response":{"id":"server_2"}}`
	waitFor(t, conn, provider.RealtimeResponseStarted)
	push <- `{"type":"error","error":{"type":"invalid_request_error","code":"conversation_already_has_active_response","message":"busy"}}`
	waitFor(t, conn, provider.RealtimeError)
	push <- `{"type":"response.done","response":{"id":"server_2","status":"completed"}}`
	expectMessages(t, received, `{"type":"response.create"}`)
}

func TestInterruptTruncatesToReceivedAudio(t *testing.T) {
	verifyNoLeaks(t)

	server, received, push := scriptedServer(t)
	conn, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()
	next(t, conn) // ready

	// One second of 24 kHz PCM16 for item_a.
	second := base64.StdEncoding.EncodeToString(make([]byte, 48_000))
	push <- `{"type":"response.created","response":{"id":"r1"}}`
	push <- `{"type":"response.output_audio.delta","response_id":"r1","item_id":"item_a","delta":"` + second + `"}`
	waitFor(t, conn, provider.RealtimeOutputAudio)

	if err := conn.Send(t.Context(), provider.RealtimeClientEvent{
		Type:   provider.RealtimeResponseCancel,
		Cursor: &provider.PlaybackCursor{ItemID: "item_a", PlayedFor: 1500 * time.Millisecond},
	}); err != nil {
		t.Fatal(err)
	}
	expectMessages(t, received,
		`{"type":"response.cancel"}`,
		`{"audio_end_ms":1000,"content_index":0,"item_id":"item_a","type":"conversation.item.truncate"}`)
}

func TestNoAutoResponseForText(t *testing.T) {
	verifyNoLeaks(t)

	server, received, _ := scriptedServer(t)
	conn, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{
		TurnDetection: provider.TurnDetection{Mode: provider.TurnModeManual, DisableAutoResponse: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()
	next(t, conn)
	if err := conn.Send(t.Context(), provider.RealtimeClientEvent{Type: provider.RealtimeTextInput, Text: "x"}); err != nil {
		t.Fatal(err)
	}
	expectMessages(t, received, `{"item":{"content":[{"text":"x","type":"input_text"}],"role":"user","type":"message"},"type":"conversation.item.create"}`)
	expectNoMessage(t, received)
}

func TestAudioDuration(t *testing.T) {
	if got := audioDuration(48_000, pcm24k); got != 1000 {
		t.Fatalf("PCM16 24 kHz = %d ms", got)
	}
	if got := audioDuration(8_000, pcmu8k); got != 1000 {
		t.Fatalf("G.711 8 kHz = %d ms", got)
	}
	if got := audioDuration(10, provider.AudioFormat{}); got != 0 {
		t.Fatalf("zero format = %d", got)
	}
}

// TestToolLoopContinuesOnce checks that parallel tool outputs are followed by a
// single response.create, whether outputs arrive before or after response.done.
func TestToolLoopContinuesOnce(t *testing.T) {
	verifyNoLeaks(t)

	received := make(chan string, 16)
	server := testServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		acceptSession(t, ctx, conn)
		sendEvent(t, ctx, conn, `{"type":"response.created","response":{"id":"resp_1"}}`)
		sendEvent(t, ctx, conn, `{"type":"response.function_call_arguments.done","response_id":"resp_1","item_id":"fc_1","call_id":"call_1","name":"a","arguments":"{\"x\":1}"}`)
		sendEvent(t, ctx, conn, `{"type":"response.function_call_arguments.done","response_id":"resp_1","item_id":"fc_2","call_id":"call_2","name":"b","arguments":""}`)
		// Wait for the first output before finishing the response.
		_, data, err := conn.Read(ctx)
		if err != nil {
			return
		}
		received <- string(data)
		sendEvent(t, ctx, conn, `{"type":"response.done","response":{"id":"resp_1","status":"completed","output":[{"type":"function_call"},{"type":"function_call"}]}}`)
		for {
			_, data, err := conn.Read(ctx)
			if err != nil {
				return
			}
			received <- string(data)
		}
	})

	conn, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()

	next(t, conn) // ready
	next(t, conn) // response started
	call1, call2 := next(t, conn), next(t, conn)
	if call1.ToolCall.ID != "call_1" || string(call1.ToolCall.Input) != `{"x":1}` || string(call2.ToolCall.Input) != `{}` {
		t.Fatalf("tool calls = %+v / %+v", call1.ToolCall, call2.ToolCall)
	}

	result := func(id string) provider.RealtimeClientEvent {
		return provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{ToolCallID: id, Output: json.RawMessage(`{"ok":true}`)}}
	}
	if err := conn.Send(t.Context(), result("call_1")); err != nil {
		t.Fatal(err)
	}
	if ev := next(t, conn); ev.FinishReason != provider.FinishToolCalls {
		t.Fatalf("finish reason = %s", ev.FinishReason)
	}
	if err := conn.Send(t.Context(), result("call_2")); err != nil {
		t.Fatal(err)
	}

	var got []string
	timeout := time.After(2 * time.Second)
	for len(got) < 3 {
		select {
		case msg := <-received:
			got = append(got, msg)
		case <-timeout:
			t.Fatalf("received only %v", got)
		}
	}
	if !strings.Contains(got[0], `"call_id":"call_1"`) || !strings.Contains(got[1], `"call_id":"call_2"`) || got[2] != `{"type":"response.create"}` {
		t.Fatalf("client events = %v", got)
	}
	select {
	case extra := <-received:
		t.Fatalf("more than one response.create: %s", extra)
	case <-time.After(100 * time.Millisecond):
	}
}

func TestRealtimeSessionToolRoundTrip(t *testing.T) {
	verifyNoLeaks(t)

	received := make(chan string, 4)
	server := testServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		acceptSession(t, ctx, conn)
		sendEvent(t, ctx, conn, `{"type":"response.created","response":{"id":"r"}}`)
		sendEvent(t, ctx, conn, `{"type":"response.function_call_arguments.done","call_id":"c","name":"weather","arguments":"{\"city\":\"Lima\"}"}`)
		sendEvent(t, ctx, conn, `{"type":"response.done","response":{"id":"r","status":"completed","output":[{"type":"function_call"}]}}`)
		for range 2 {
			_, data, err := conn.Read(ctx)
			if err != nil {
				return
			}
			received <- string(data)
		}
		drain(ctx, conn)
	})

	weather := goai.Tool{
		Name: "weather",
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			return "sunny", nil
		},
	}
	session, err := goai.OpenRealtimeSession(t.Context(), testModel(server.URL), goai.WithRealtimeTools(weather))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close() }()

	for _, want := range []string{
		`{"item":{"call_id":"c","output":"{\"text\":\"sunny\"}","type":"function_call_output"},"type":"conversation.item.create"}`,
		`{"type":"response.create"}`,
	} {
		select {
		case got := <-received:
			if got != want {
				t.Fatalf("client sent %s\nwant        %s", got, want)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("server did not receive %s", want)
		}
	}
}

func TestConnectionErrors(t *testing.T) {
	verifyNoLeaks(t)

	t.Run("http error on upgrade", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}`))
		}))
		defer server.Close()
		_, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{})
		var apiErr *goai.APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusUnauthorized {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("error during setup", func(t *testing.T) {
		server := testServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			readEvent(t, ctx, conn)
			sendEvent(t, ctx, conn, `{"type":"error","error":{"type":"invalid_request_error","code":"invalid_value","message":"Invalid voice"}}`)
			drain(ctx, conn)
		})
		_, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{Voice: "nope"})
		if err == nil || !strings.Contains(err.Error(), "Invalid voice") {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("closed during setup", func(t *testing.T) {
		server := testServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			readEvent(t, ctx, conn)
			_ = conn.Close(websocket.StatusNormalClosure, "")
		})
		if _, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
			t.Fatal("expected setup failure")
		}
	})

	t.Run("abnormal close mid session", func(t *testing.T) {
		server := testServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			acceptSession(t, ctx, conn)
			_ = conn.Close(websocket.StatusPolicyViolation, "session expired")
		})
		conn, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{})
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = conn.Close() }()
		for range conn.Events() {
		}
		if err := conn.Err(); err == nil || !strings.Contains(err.Error(), "session expired") {
			t.Fatalf("Err() = %v", err)
		}
	})

	t.Run("normal close mid session", func(t *testing.T) {
		server := testServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			acceptSession(t, ctx, conn)
			_ = conn.Close(websocket.StatusNormalClosure, "")
		})
		conn, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{})
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = conn.Close() }()
		for range conn.Events() {
		}
		if err := conn.Err(); err != nil {
			t.Fatalf("Err() = %v", err)
		}
	})

	t.Run("malformed event", func(t *testing.T) {
		server := testServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			acceptSession(t, ctx, conn)
			sendEvent(t, ctx, conn, `{nope`)
			drain(ctx, conn)
		})
		conn, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{})
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = conn.Close() }()
		for range conn.Events() {
		}
		if conn.Err() == nil {
			t.Fatal("expected a decode error")
		}
	})

	t.Run("bad audio base64", func(t *testing.T) {
		evs := newTestConn().translate(serverEvent{Type: "response.output_audio.delta", Delta: "!!"})
		if len(evs) != 1 || evs[0].Type != provider.RealtimeError {
			t.Fatalf("events = %+v", evs)
		}
	})
}

func TestFinishReason(t *testing.T) {
	tests := []struct {
		resp wireResponse
		want provider.FinishReason
	}{
		{wireResponse{Status: "completed"}, provider.FinishStop},
		{wireResponse{Status: "completed", Output: []wireOutputItem{{Type: "function_call"}}}, provider.FinishToolCalls},
		{wireResponse{Status: "cancelled", StatusDetails: &wireStatusDetails{Reason: "turn_detected"}}, provider.FinishInterrupted},
		{wireResponse{Status: "incomplete", StatusDetails: &wireStatusDetails{Reason: "max_output_tokens"}}, provider.FinishLength},
		{wireResponse{Status: "incomplete", StatusDetails: &wireStatusDetails{Reason: "content_filter"}}, provider.FinishContentFilter},
		{wireResponse{Status: "failed"}, provider.FinishError},
	}
	for _, tt := range tests {
		if got := finishReason(&tt.resp); got != tt.want {
			t.Errorf("finishReason(%+v) = %s, want %s", tt.resp, got, tt.want)
		}
	}

	c := newTestConn()
	failed := c.responseDone(&wireResponse{ID: "r", Status: "failed", StatusDetails: &wireStatusDetails{Error: &wireError{Type: "server_error", Message: "boom"}}})
	if len(failed) != 2 || failed[1].Type != provider.RealtimeError || !strings.Contains(failed[1].Error.Error(), "boom") {
		t.Fatalf("failed response events = %+v", failed)
	}
	if evs := c.responseDone(nil); len(evs) != 1 || evs[0].FinishReason != provider.FinishStop {
		t.Fatalf("nil response events = %+v", evs)
	}
	if err := c.serverError(nil); err == nil {
		t.Fatal("serverError(nil) must still report an error")
	}
}

func TestCapabilitiesAndURL(t *testing.T) {
	model := testModel("https://api.openai.com")
	if model.ModelID() != "gpt-realtime-2.1" {
		t.Fatalf("ModelID = %q", model.ModelID())
	}
	caps := provider.RealtimeCapabilitiesOf(model)
	if !caps.OutputAudioTruncation || !caps.ExplicitResponseRequest || caps.ClientVAD || caps.SessionResumption {
		t.Fatalf("capabilities = %+v", caps)
	}
	if got, _ := realtimeURL("https://res.openai.azure.com/openai/v1/", "my deploy"); got != "wss://res.openai.azure.com/openai/v1/realtime?model=my+deploy" {
		t.Fatalf("realtimeURL = %s", got)
	}
}

func FuzzTranslate(f *testing.F) {
	for _, seed := range []string{
		`{"type":"response.output_audio.delta","delta":"AQID"}`,
		`{"type":"response.function_call_arguments.done","call_id":"c","arguments":"{"}`,
		`{"type":"response.done","response":{"status":"incomplete","status_details":{"reason":"content_filter"}}}`,
		`{"type":"rate_limits.updated","rate_limits":[{"reset_seconds":-1}]}`,
		`{"type":"error"}`,
		`{"type":"response.created"}`,
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(_ *testing.T, data []byte) {
		var ev serverEvent
		if json.Unmarshal(data, &ev) != nil {
			return
		}
		c := newTestConn()
		c.outputTr = true
		_ = c.translate(ev)
		_ = c.finishResponse()
	})
}
