package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
	"go.uber.org/goleak"
)

func liveNoLeaks(t *testing.T) {
	t.Helper()
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })
}

// liveServer is a minimal GPT-Live endpoint; handle owns the server socket.
func liveServer(t *testing.T, handle func(ctx context.Context, conn *websocket.Conn, r *http.Request)) *httptest.Server {
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

func liveRead(ctx context.Context, conn *websocket.Conn) (map[string]any, string, bool) {
	_, data, err := conn.Read(ctx)
	if err != nil {
		return nil, "", false
	}
	var ev map[string]any
	_ = json.Unmarshal(data, &ev)
	return ev, string(data), true
}

func liveWrite(ctx context.Context, conn *websocket.Conn, ev string) {
	_ = conn.Write(ctx, websocket.MessageText, []byte(ev))
}

// liveAccept answers session.start and returns its session object.
func liveAccept(ctx context.Context, conn *websocket.Conn) map[string]any {
	ev, _, ok := liveRead(ctx, conn)
	if !ok || ev["type"] != "session.start" {
		return nil
	}
	liveWrite(ctx, conn, `{"type":"session.started","session":{"id":"sess_live"}}`)
	session, _ := ev["session"].(map[string]any)
	return session
}

// liveRelay forwards every client message and answers session.close.
func liveRelay(ctx context.Context, conn *websocket.Conn, received chan<- string, closedUsage string) {
	for {
		ev, raw, ok := liveRead(ctx, conn)
		if !ok {
			return
		}
		received <- raw
		if ev["type"] == "session.close" && closedUsage != "" {
			liveWrite(ctx, conn, `{"type":"session.closed","reason":"client","usage":`+closedUsage+`}`)
			_ = conn.Close(websocket.StatusNormalClosure, "")
			return
		}
	}
}

func liveNext(t *testing.T, conn provider.RealtimeConnection) provider.RealtimeEvent {
	t.Helper()
	select {
	case ev, ok := <-conn.Events():
		if !ok {
			t.Fatalf("events closed early; Err() = %v", conn.Err())
		}
		return ev
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for a live event")
		return provider.RealtimeEvent{}
	}
}

func liveExpect(t *testing.T, received chan string, want ...string) {
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

func TestLive_SessionStartWithResponsesDelegation(t *testing.T) {
	liveNoLeaks(t)

	type handshake struct {
		path, auth string
		session    map[string]any
	}
	got := make(chan handshake, 1)
	server := liveServer(t, func(ctx context.Context, conn *websocket.Conn, r *http.Request) {
		session := liveAccept(ctx, conn)
		got <- handshake{r.URL.Path, r.Header.Get("Authorization"), session}
		liveRelay(ctx, conn, make(chan string, 8), `{"seconds":1}`)
	})

	delegation := LiveResponsesDelegation("gpt-6-luna")
	backend := delegation["delegation"].(map[string]any)["responses"].(map[string]any)
	backend["instructions"] = "Backend prompt."
	backend["tools"] = []any{map[string]any{"type": "web_search"}}

	model := Live("gpt-live-1", WithAPIKey("sk-live"), WithBaseURL(server.URL+"/v1"))
	conn, err := model.OpenRealtime(t.Context(), provider.RealtimeParams{
		System:          "Be warm.",
		Voice:           "marin",
		InputAudio:      livePCM16k,
		ToolChoice:      "auto",
		MaxOutputTokens: 400,
		Tools:           []provider.ToolDefinition{{Name: "lookup", Description: "Looks up", InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)}},
		ProviderOptions: delegation,
	})
	if err != nil {
		t.Fatal(err)
	}
	if ev := liveNext(t, conn); ev.Type != provider.RealtimeSessionReady || ev.SessionID != "sess_live" {
		t.Fatalf("ready event = %+v", ev)
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}

	h := <-got
	if h.path != "/v1/live/sessions" || h.auth != "Bearer sk-live" {
		t.Errorf("handshake = %s %q", h.path, h.auth)
	}
	body, _ := json.Marshal(h.session)
	for _, want := range []string{
		`"model":"gpt-live-1"`,
		`"instructions":"Be warm."`,
		`"audio":{"format":{"rate":16000,"type":"audio/pcm"},"output":{"voice":"marin"}}`,
		`"type":"responses"`,
		`"model":"gpt-6-luna"`,
		`"instructions":"Backend prompt."`,
		`{"type":"web_search"}`,
		`"name":"lookup"`,
		`"tool_choice":"auto"`,
		`"max_output_tokens":400`,
	} {
		if !strings.Contains(string(body), want) {
			t.Errorf("session.start missing %s\n%s", want, body)
		}
	}
	if _, ok := backend["tool_choice"]; ok {
		t.Error("caller's delegation map was mutated")
	}
}

func TestLive_ClientDelegation(t *testing.T) {
	liveNoLeaks(t)

	received := make(chan string, 16)
	server := liveServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		liveAccept(ctx, conn)
		liveWrite(ctx, conn, `{"type":"session.input_transcript.delta","delta":"book a table","start_ms":0,"end_ms":900}`)
		liveWrite(ctx, conn, `{"type":"session.delegation.created","offset_ms":1000,"delegation":{"id":"item_1","type":"delegation","target":"client"}}`)
		liveWrite(ctx, conn, `{"type":"session.output_audio.delta","delta":"AQID"}`)
		liveWrite(ctx, conn, `{"type":"session.output_transcript.delta","delta":"Let me check."}`)
		liveWrite(ctx, conn, `{"type":"session.usage.updated","usage":{"seconds":4.5},"context_window":{"usage_ratio":0.1}}`)
		liveWrite(ctx, conn, `{"type":"session.commentary.appended","client_event_id":"x"}`)
		liveWrite(ctx, conn, `{"type":"error","error":{"type":"invalid_request_error","code":"unknown_delegation","message":"no such delegation"}}`)
		liveRelay(ctx, conn, received, `{"seconds":9.25}`)
	})

	conn, err := Live("gpt-live-1", WithAPIKey("k"), WithBaseURL(server.URL+"/v1")).OpenRealtime(t.Context(), provider.RealtimeParams{
		InputTranscription:  provider.TranscriptionConfig{Enabled: true},
		OutputTranscription: provider.TranscriptionConfig{Enabled: true},
	})
	if err != nil {
		t.Fatal(err)
	}

	want := []provider.RealtimeEventType{
		provider.RealtimeSessionReady,
		provider.RealtimeInputTranscript,
		provider.RealtimeDelegationCreated,
		provider.RealtimeOutputAudio,
		provider.RealtimeOutputTranscript,
		provider.RealtimeUsageUpdated,
		provider.RealtimeError,
	}
	events := make([]provider.RealtimeEvent, len(want))
	for i := range want {
		events[i] = liveNext(t, conn)
		if events[i].Type != want[i] {
			t.Fatalf("event %d = %s, want %s", i, events[i].Type, want[i])
		}
	}
	if d := events[2].Delegation; d == nil || d.ID != "item_1" || d.Target != "client" || d.Offset != time.Second {
		t.Errorf("delegation = %+v", d)
	}
	if ev := events[3]; string(ev.Audio) != "\x01\x02\x03" || ev.AudioFormat != livePCM24k {
		t.Errorf("audio = %+v", ev)
	}
	if ev := events[5]; ev.UsageDuration != 4500*time.Millisecond {
		t.Errorf("usage duration = %s", ev.UsageDuration)
	}
	if err := events[6].Error; err == nil || !strings.Contains(err.Error(), "unknown_delegation") {
		t.Errorf("error event = %v", err)
	}

	ctx := t.Context()
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioAppend, Audio: []byte{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}
	liveExpect(t, received, `{"audio":"AQID","type":"session.input_audio.append"}`)
	for _, c := range []struct {
		content provider.RealtimeContext
		want    string
	}{
		{provider.RealtimeContext{Kind: provider.RealtimeContextCommentary, DelegationID: "item_1", Text: "Booked for 8 PM."}, `{"content":"Booked for 8 PM.","delegation_id":"item_1","type":"session.commentary.append"}`},
		{provider.RealtimeContext{Kind: provider.RealtimeContextThinking, Text: "User is on the checkout page."}, `{"content":"User is on the checkout page.","delegation_id":null,"type":"session.thinking.append"}`},
		{provider.RealtimeContext{Kind: provider.RealtimeContextInstructions, Text: "Stop."}, `{"content":"Stop.","delegation_id":null,"type":"session.instructions.append"}`},
	} {
		content := c.content
		if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeContextAppend, Context: &content}); err != nil {
			t.Fatal(err)
		}
		liveExpect(t, received, c.want)
	}

	for _, ev := range []provider.RealtimeClientEvent{
		{Type: provider.RealtimeTextInput, Text: "x"},
		{Type: provider.RealtimeResponseCreate},
		{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{ToolCallID: "c"}},
		{Type: provider.RealtimeResponseCancel},
		{Type: provider.RealtimeInputAudioCommit},
		{Type: provider.RealtimeInputActivityStart},
	} {
		if err := conn.Send(ctx, ev); !errors.Is(err, provider.ErrRealtimeUnsupported) {
			t.Errorf("Send(%s) = %v, want ErrRealtimeUnsupported", ev.Type, err)
		}
	}
	for _, ev := range []provider.RealtimeClientEvent{
		{Type: provider.RealtimeInputAudioAppend},
		{Type: provider.RealtimeContextAppend},
		{Type: provider.RealtimeContextAppend, Context: &provider.RealtimeContext{Kind: "shout", Text: "x"}},
		{Type: provider.RealtimeToolResultEvent},
	} {
		if err := conn.Send(ctx, ev); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
			t.Errorf("Send(%s) = %v, want ErrInvalidRealtimeConfig", ev.Type, err)
		}
	}

	// Close sends session.close and still delivers the final usage.
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
	liveExpect(t, received, `{"type":"session.close"}`)
	var final provider.RealtimeEvent
	for ev := range conn.Events() {
		final = ev
	}
	if final.Type != provider.RealtimeUsageUpdated || final.UsageDuration != 9250*time.Millisecond {
		t.Errorf("final event = %+v", final)
	}
	if conn.Err() != nil {
		t.Errorf("Err after close = %v", conn.Err())
	}
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioAppend, Audio: []byte{1}}); !errors.Is(err, provider.ErrRealtimeSessionClosed) {
		t.Errorf("Send after Close = %v", err)
	}
}

func TestLive_TranscriptsOnlyWhenEnabled(t *testing.T) {
	c := &liveConn{pending: map[string]bool{}}
	for _, typ := range []string{"session.input_transcript.delta", "session.output_transcript.delta"} {
		if evs := c.translate(liveServerEvent{Type: typ, Delta: "x"}); len(evs) != 0 {
			t.Errorf("%s emitted without transcription enabled: %+v", typ, evs)
		}
	}
}

// TestLive_ResponsesToolLoop runs Responses delegation through the goai
// session: both function calls execute, their outputs go back as
// response.item.create, and a single response.create follows.
func TestLive_ResponsesToolLoop(t *testing.T) {
	liveNoLeaks(t)

	received := make(chan string, 16)
	server := liveServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		liveAccept(ctx, conn)
		liveWrite(ctx, conn, `{"type":"session.delegation.created","offset_ms":10,"delegation":{"id":"item_9","type":"delegation","target":"responses","response_id":"resp_1"}}`)
		liveWrite(ctx, conn, `{"type":"response.event","delegation_id":"item_9","event":{"type":"response.output_item.done","item":{"type":"web_search_call","id":"ws_1"}}}`)
		liveWrite(ctx, conn, `{"type":"response.event","delegation_id":"item_9","event":{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_a","name":"lookup","arguments":"{\"q\":\"a\"}"}}}`)
		liveWrite(ctx, conn, `{"type":"response.event","delegation_id":"item_9","event":{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_b","name":"lookup","arguments":""}}}`)
		liveWrite(ctx, conn, `{"type":"response.event","delegation_id":"item_9","event":{"type":"response.completed","response":{"id":"resp_1","output":[]}}}`)
		liveRelay(ctx, conn, received, `{"seconds":2}`)
	})

	var calls []string
	lookup := goai.Tool{
		Name: "lookup",
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			calls = append(calls, string(input))
			return `{"found":true}`, nil
		},
	}
	session, err := goai.OpenRealtimeSession(t.Context(),
		Live("gpt-live-1", WithAPIKey("k"), WithBaseURL(server.URL+"/v1")),
		goai.WithRealtimeTools(lookup),
		goai.WithRealtimeSequentialTools(true),
		goai.WithRealtimeProviderOptions(LiveResponsesDelegation("gpt-6-luna")),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Tool goroutines finish in any order; both outputs must precede the single
	// response.create.
	outputs := map[string]bool{}
	for range 2 {
		select {
		case got := <-received:
			if !strings.Contains(got, `"type":"function_call_output"`) || !strings.Contains(got, `"type":"response.item.create"`) {
				t.Fatalf("expected a tool output, got %s", got)
			}
			outputs[got] = true
		case <-time.After(2 * time.Second):
			t.Fatal("tool outputs never arrived")
		}
	}
	for _, id := range []string{"call_a", "call_b"} {
		want := `{"item":{"call_id":"` + id + `","output":"{\"found\":true}","type":"function_call_output"},"type":"response.item.create"}`
		if !outputs[want] {
			t.Errorf("missing output for %s: %v", id, outputs)
		}
	}
	liveExpect(t, received, `{"type":"response.create"}`)
	select {
	case extra := <-received:
		t.Fatalf("unexpected client event: %s", extra)
	case <-time.After(100 * time.Millisecond):
	}

	ctx := t.Context()
	if err := session.SendText(ctx, "Order A0042"); err != nil {
		t.Fatal(err)
	}
	liveExpect(t, received,
		`{"item":{"content":[{"text":"Order A0042","type":"input_text"}],"role":"user","type":"message"},"type":"response.item.create"}`,
		`{"type":"response.create"}`)
	if err := session.RequestResponse(ctx); err != nil {
		t.Fatal(err)
	}
	liveExpect(t, received, `{"type":"response.create"}`)

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	slices.Sort(calls)
	if strings.Join(calls, ",") != `{"q":"a"},{}` {
		t.Errorf("tool inputs = %v", calls)
	}
}

func TestLive_BackendErrors(t *testing.T) {
	c := &liveConn{pending: map[string]bool{}}
	for _, raw := range []string{
		`{"type":"response.failed","response":{"id":"r","error":{"code":"server_error","message":"backend down"}}}`,
		`{"type":"error","code":"tool_outputs_required","message":"missing outputs"}`,
		`{"type":"error","error":{"code":"x","message":"nested"}}`,
	} {
		evs := c.translateBackend(json.RawMessage(raw))
		if len(evs) != 1 || evs[0].Type != provider.RealtimeError {
			t.Errorf("translateBackend(%s) = %+v", raw, evs)
		}
	}
	for _, raw := range []string{``, `{bad`, `{"type":"response.output_text.delta","delta":"x"}`, `{"type":"response.output_item.done"}`} {
		if evs := c.translateBackend(json.RawMessage(raw)); len(evs) != 0 {
			t.Errorf("translateBackend(%q) = %+v", raw, evs)
		}
	}
	if evs := c.translate(liveServerEvent{Type: "session.output_audio.delta", Delta: "!!"}); len(evs) != 1 || evs[0].Type != provider.RealtimeError {
		t.Errorf("bad audio = %+v", evs)
	}
	if evs := c.translate(liveServerEvent{Type: "session.delegation.created"}); len(evs) != 0 {
		t.Errorf("delegation without payload = %+v", evs)
	}
	if evs := c.translate(liveServerEvent{Type: "session.usage.updated"}); len(evs) != 0 {
		t.Errorf("usage without payload = %+v", evs)
	}
	if c.outputSent("unknown") {
		t.Error("an unknown call must not trigger response.create")
	}
	if err := liveError(nil); err == nil {
		t.Error("liveError(nil) must still report an error")
	}
}

func TestLive_SessionValidation(t *testing.T) {
	model := Live("gpt-live-1", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1/v1"))
	pcm8k := provider.AudioFormat{Encoding: provider.AudioPCM16, SampleRate: 8_000, Channels: 1}
	tests := []struct {
		name   string
		params provider.RealtimeParams
		want   error
	}{
		{"unsupported rate", provider.RealtimeParams{InputAudio: pcm8k}, provider.ErrUnsupportedAudioFormat},
		{"different formats", provider.RealtimeParams{InputAudio: livePCM16k, OutputAudio: livePCM24k}, provider.ErrUnsupportedAudioFormat},
		{"manual turns", provider.RealtimeParams{TurnDetection: provider.TurnDetection{Mode: provider.TurnModeManual}}, provider.ErrRealtimeUnsupported},
		{"no interruption", provider.RealtimeParams{TurnDetection: provider.TurnDetection{DisableInterruption: true}}, provider.ErrRealtimeUnsupported},
		{"transcription language", provider.RealtimeParams{InputTranscription: provider.TranscriptionConfig{Enabled: true, Language: "es"}}, provider.ErrRealtimeUnsupported},
		{"tools without responses", provider.RealtimeParams{Tools: []provider.ToolDefinition{{Name: "t"}}}, provider.ErrRealtimeUnsupported},
		{"responses without backend", provider.RealtimeParams{ProviderOptions: map[string]any{"delegation": map[string]any{"type": "responses"}}}, provider.ErrInvalidRealtimeConfig},
		{"hosted tool def", provider.RealtimeParams{Tools: []provider.ToolDefinition{{Name: "w", ProviderDefinedType: "web_search"}}, ProviderOptions: LiveResponsesDelegation("m")}, provider.ErrRealtimeUnsupported},
		{"bad schema", provider.RealtimeParams{Tools: []provider.ToolDefinition{{Name: "t", InputSchema: json.RawMessage(`{bad`)}}, ProviderOptions: LiveResponsesDelegation("m")}, provider.ErrInvalidRealtimeConfig},
		{"override model", provider.RealtimeParams{ProviderOptions: map[string]any{"model": "x"}}, provider.ErrInvalidRealtimeConfig},
		{"common validation", provider.RealtimeParams{MaxOutputTokens: -1}, provider.ErrInvalidRealtimeConfig},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := model.OpenRealtime(t.Context(), tt.params); !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
		})
	}

	t.Setenv("OPENAI_API_KEY", "")
	if _, err := Live("m").OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected a missing key error")
	}
	failing := provider.CachedTokenSource(func(context.Context) (*provider.Token, error) { return nil, context.DeadlineExceeded })
	if _, err := Live("m", WithTokenSource(failing)).OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected a token source error")
	}
	if _, err := Live("m", WithAPIKey("k"), WithBaseURL("ftp://x")).OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected unsupported scheme error")
	}
	for f, want := range map[provider.AudioFormat]string{
		livePCMU: `{"rate":8000,"type":"audio/pcmu"}`,
		livePCMA: `{"rate":8000,"type":"audio/pcma"}`,
	} {
		_, wire, err := liveFormat(f, f)
		got, _ := json.Marshal(wire)
		if err != nil || string(got) != want {
			t.Errorf("liveFormat(%+v) = %s, %v", f, got, err)
		}
	}
	if v := toolChoiceValue("lookup"); v.(map[string]any)["name"] != "lookup" {
		t.Errorf("named tool choice = %v", v)
	}
	if LiveClientDelegation()["delegation"].(map[string]any)["type"] != "client" {
		t.Error("LiveClientDelegation type")
	}
}

func TestLive_ConnectionErrors(t *testing.T) {
	liveNoLeaks(t)

	t.Run("http error on upgrade", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"message":"Incorrect API key provided"}}`))
		}))
		defer server.Close()
		_, err := Live("gpt-live-1", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
		var apiErr *goai.APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusUnauthorized {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("error during start", func(t *testing.T) {
		server := liveServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			liveRead(ctx, conn)
			liveWrite(ctx, conn, `{"type":"error","error":{"type":"invalid_request_error","code":"responses_validation_failed","message":"bad backend"}}`)
			liveRelay(ctx, conn, make(chan string, 4), "")
		})
		_, err := Live("gpt-live-1", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
		if err == nil || !strings.Contains(err.Error(), "bad backend") {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("closed during start", func(t *testing.T) {
		server := liveServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			liveRead(ctx, conn)
			_ = conn.Close(websocket.StatusNormalClosure, "")
		})
		if _, err := Live("gpt-live-1", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
			t.Fatal("expected start failure")
		}
	})

	t.Run("abnormal close mid session", func(t *testing.T) {
		server := liveServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			liveAccept(ctx, conn)
			_ = conn.Close(websocket.StatusPolicyViolation, "session expired")
		})
		conn, err := Live("gpt-live-1", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
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

	t.Run("malformed event", func(t *testing.T) {
		server := liveServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			liveAccept(ctx, conn)
			liveWrite(ctx, conn, `{nope`)
			liveRelay(ctx, conn, make(chan string, 4), "")
		})
		conn, err := Live("gpt-live-1", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
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

	t.Run("close times out without session.closed", func(t *testing.T) {
		previous := liveCloseTimeout
		liveCloseTimeout = 50 * time.Millisecond
		t.Cleanup(func() { liveCloseTimeout = previous })

		server := liveServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			liveAccept(ctx, conn)
			liveRelay(ctx, conn, make(chan string, 4), "")
		})
		conn, err := Live("gpt-live-1", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		if err := conn.Close(); err != nil {
			t.Fatal(err)
		}
		if elapsed := time.Since(start); elapsed > 2*time.Second {
			t.Fatalf("Close took %s", elapsed)
		}
	})
}

func TestLive_Capabilities(t *testing.T) {
	model := Live("gpt-live-1")
	if model.ModelID() != "gpt-live-1" {
		t.Fatalf("ModelID = %q", model.ModelID())
	}
	caps := provider.RealtimeCapabilitiesOf(model)
	if !caps.Delegation || !caps.BargeIn || caps.ManualTurnControl || caps.OutputAudioTruncation || len(caps.SupportedInputAudio) != 4 {
		t.Fatalf("capabilities = %+v", caps)
	}
	if got, _ := liveURL("https://api.openai.com/v1/"); got != "wss://api.openai.com/v1/live/sessions" {
		t.Fatalf("liveURL = %s", got)
	}
}

func FuzzLiveTranslate(f *testing.F) {
	for _, seed := range []string{
		`{"type":"session.output_audio.delta","delta":"AQID"}`,
		`{"type":"session.delegation.created","offset_ms":-5,"delegation":{"id":"d"}}`,
		`{"type":"response.event","event":{"type":"response.output_item.done","item":{"type":"function_call","call_id":"c","arguments":"{"}}}`,
		`{"type":"response.event","event":"not an object"}`,
		`{"type":"session.closed","usage":{"seconds":-1}}`,
		`{"type":"error"}`,
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(_ *testing.T, data []byte) {
		var ev liveServerEvent
		if json.Unmarshal(data, &ev) != nil {
			return
		}
		c := &liveConn{pending: map[string]bool{}, inputTr: true, outputTr: true}
		_ = c.translate(ev)
	})
}
