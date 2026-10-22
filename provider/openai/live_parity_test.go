package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/zendev-sh/goai/provider"
)

func TestLiveProviderOptions(t *testing.T) {
	assertJSON(t, LiveProviderOptions(LiveConfig{
		Store:   ptrTo(true),
		VoiceID: "voice_1",
		Input:   []LiveInitialItem{{Role: "developer", Text: "Greet first."}, {Role: "assistant", Text: "Hi there!"}},
		Triggers: []LiveDecisionTrigger{{
			EventType: "session.input_transcript.delta",
			Questions: []LiveDecisionQuestion{{Kind: "predicate", Name: "angry", Instructions: "Is the caller angry?"}},
		}},
		Decisions: &LiveDecisions{Model: "gpt-decide", Questions: []LiveDecisionQuestion{
			{Kind: "choice", Instructions: "Which department?", Choices: []LiveDecisionChoice{{Value: "sales", Description: "Buying"}, {Value: true}}},
			{Kind: "score", Name: "urgency", Instructions: "How urgent?", Levels: []LiveDecisionLevel{{Label: "low"}, {Label: "high", Description: "Now"}}},
		}},
	}), `{
		"store":true,
		"audio":{"output":{"voice":{"id":"voice_1"}}},
		"input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"Greet first."}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hi there!"}]}],
		"triggers":[{"type":"decisions","event_type":"session.input_transcript.delta",
			"decisions":{"questions":[{"type":"predicate","name":"angry","instructions":"Is the caller angry?"}]}}],
		"delegation":{"type":"decisions","decisions":{"model":"gpt-decide","questions":[
			{"type":"choice","instructions":"Which department?","choices":[{"value":"sales","description":"Buying"},{"value":true}]},
			{"type":"score","name":"urgency","instructions":"How urgent?","levels":[{"label":"low"},{"label":"high","description":"Now"}]}]}}}`)
	assertJSON(t, LiveProviderOptions(LiveConfig{}), `{}`)
}

func TestLiveRawCommands(t *testing.T) {
	item, err := LiveCreateResponseItem(map[string]any{"type": "message", "role": "user"})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		got  json.RawMessage
		want string
	}{
		{LiveMuteInput(), `{"type":"session.input_audio.mute"}`},
		{LiveUnmuteInput(), `{"type":"session.input_audio.unmute"}`},
		{LivePlayFixture([]byte{1, 2}), `{"type":"session.output_audio.fixture.play","audio":"AQI="}`},
		{LiveCreateDecision(), `{"type":"decision.create"}`},
		{LiveDeleteDecisionItem("d1"), `{"type":"decision.item.delete","item_id":"d1"}`},
		{item, `{"type":"response.item.create","item":{"type":"message","role":"user"}}`},
	} {
		assertJSON(t, json.RawMessage(c.got), c.want)
	}
	if _, err := LiveCreateResponseItem(map[string]any{"bad": make(chan int)}); err == nil {
		t.Error("an item that cannot encode must fail")
	}
}

func TestLiveHostedToolsInResponsesDelegation(t *testing.T) {
	session, _, kind, err := buildLiveSession("gpt-live-1", provider.RealtimeParams{
		Tools: []provider.ToolDefinition{
			Tools.WebSearch(),
			Tools.MCP("docs", WithMCPServerURL("https://mcp.example.com")),
			{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		ProviderOptions: LiveResponsesDelegation("gpt-6-luna"),
	})
	if err != nil || kind != "responses" {
		t.Fatalf("kind %q, err %v", kind, err)
	}
	tools := session["delegation"].(map[string]any)["responses"].(map[string]any)["tools"].([]any)
	if len(tools) != 3 || tools[0].(map[string]any)["type"] != "web_search" || tools[1].(map[string]any)["server_label"] != "docs" ||
		tools[2].(map[string]any)["type"] != "function" {
		t.Fatalf("tools = %v", tools)
	}
}

// liveModeServer records the path of the upgrade and forwards client messages.
func liveModeServer(t *testing.T, handle func(ctx context.Context, conn *websocket.Conn)) (*httptest.Server, <-chan string) {
	t.Helper()
	paths := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths <- r.URL.Path
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		handle(r.Context(), conn)
	}))
	t.Cleanup(server.Close)
	return server, paths
}

func TestLiveFork(t *testing.T) {
	liveNoLeaks(t)

	started := make(chan map[string]any, 1)
	received := make(chan string, 8)
	server, paths := liveModeServer(t, func(ctx context.Context, conn *websocket.Conn) {
		ev, _, ok := liveRead(ctx, conn)
		if !ok {
			return
		}
		started <- ev
		liveWrite(ctx, conn, `{"type":"session.started","session":{"id":"sess_fork","delegation":{"type":"responses"},"audio":{"format":{"type":"audio/pcm","rate":16000}}}}`)
		liveWrite(ctx, conn, `{"type":"session.output_audio.delta","delta":"AQI="}`)
		liveRelay(ctx, conn, received, `{"seconds":1}`)
	})

	conn, err := LiveFork("sess_src", WithAPIKey("k"), WithBaseURL(server.URL+"/v1")).OpenRealtime(t.Context(), provider.RealtimeParams{
		Tools:           []provider.ToolDefinition{{Name: "lookup"}},
		ProviderOptions: map[string]any{"store": false},
	})
	if err != nil {
		t.Fatal(err)
	}
	if p := <-paths; p != "/v1/live/sessions/sess_src/fork" {
		t.Errorf("path = %s", p)
	}
	ev := <-started
	if ev["type"] != "session.start" {
		t.Fatalf("first message = %v", ev)
	}
	assertJSON(t, ev["session"], `{"store":false,"delegation":{"type":"responses","responses":{"tools":[
		{"type":"function","name":"lookup","description":"","parameters":{"type":"object","properties":{}}}]}}}`)
	if ev := liveNext(t, conn); ev.Type != provider.RealtimeSessionReady || ev.SessionID != "sess_fork" {
		t.Fatalf("ready = %+v", ev)
	}
	if ev := liveNext(t, conn); ev.AudioFormat != livePCM16k {
		t.Fatalf("a fork inherits the source format: %+v", ev.AudioFormat)
	}
	// The inherited Responses delegation enables backend commands.
	if err := conn.Send(t.Context(), provider.RealtimeClientEvent{Type: provider.RealtimeResponseCreate}); err != nil {
		t.Fatal(err)
	}
	if got := <-received; got != `{"type":"response.create"}` {
		t.Fatalf("sent %s", got)
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}

	for _, params := range []provider.RealtimeParams{{System: "x"}, {Voice: "marin"}} {
		if _, err := LiveFork("s", WithAPIKey("k")).OpenRealtime(t.Context(), params); !errors.Is(err, provider.ErrRealtimeUnsupported) {
			t.Errorf("fork %+v: %v", params, err)
		}
	}
	model := LiveFork("sess_src")
	if model.ModelID() != "sess_src" {
		t.Errorf("ModelID = %s", model.ModelID())
	}
}

func TestLiveSideband(t *testing.T) {
	liveNoLeaks(t)

	mirrored := `{"type":"session.input_audio.append","audio":"AQI="}`
	received := make(chan string, 8)
	server, paths := liveModeServer(t, func(ctx context.Context, conn *websocket.Conn) {
		liveWrite(ctx, conn, mirrored)
		liveRelay(ctx, conn, received, `{"seconds":3}`)
	})

	conn, err := LiveSideband("sess_web", WithAPIKey("k"), WithBaseURL(server.URL+"/v1")).OpenRealtime(t.Context(), provider.RealtimeParams{OutputAudio: livePCMU})
	if err != nil {
		t.Fatal(err)
	}
	if p := <-paths; p != "/v1/live/sessions/sess_web/attach" {
		t.Errorf("path = %s", p)
	}
	if ev := liveNext(t, conn); ev.Type != provider.RealtimeSessionReady || ev.SessionID != "sess_web" {
		t.Fatalf("ready = %+v", ev)
	}
	if ev := liveNext(t, conn); ev.Type != provider.RealtimeProviderEvent || string(ev.Raw) != mirrored {
		t.Fatalf("mirrored input audio = %+v", ev)
	}
	ctx := t.Context()
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioAppend, Audio: []byte{1}}); !errors.Is(err, provider.ErrRealtimeUnsupported) {
		t.Errorf("audio on a sideband: %v", err)
	}
	// The delegation is unknown on a sideband, so the server decides.
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeTextInput, Text: "hi"}); err != nil {
		t.Fatal(err)
	}
	if got := <-received; !strings.Contains(got, `"response.item.create"`) {
		t.Errorf("sent %s", got)
	}
	<-received // response.create
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeRawEvent, Raw: LiveMuteInput()}); err != nil {
		t.Fatal(err)
	}
	if got := <-received; got != `{"type":"session.input_audio.mute"}` {
		t.Errorf("sent %s", got)
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}

	for _, params := range []provider.RealtimeParams{
		{System: "x"}, {Tools: []provider.ToolDefinition{{Name: "t"}}}, {ProviderOptions: map[string]any{"store": true}},
	} {
		if _, err := LiveSideband("s", WithAPIKey("k")).OpenRealtime(ctx, params); !errors.Is(err, provider.ErrRealtimeUnsupported) {
			t.Errorf("sideband %+v: %v", params, err)
		}
	}
	if _, err := LiveSideband("s", WithAPIKey("k")).OpenRealtime(ctx, provider.RealtimeParams{InputAudio: livePCM24k, OutputAudio: livePCMU}); !errors.Is(err, provider.ErrUnsupportedAudioFormat) {
		t.Errorf("mismatched formats: %v", err)
	}
	if caps := provider.RealtimeCapabilitiesOf(LiveSideband("s")); caps.InputAudio || !caps.OutputAudio {
		t.Errorf("sideband capabilities = %+v", caps)
	}
}

func TestLiveEncodeByDelegation(t *testing.T) {
	encode := func(delegation string, ev provider.RealtimeClientEvent) (string, error) {
		t.Helper()
		c := &liveConn{delegation: delegation, pending: map[string]bool{}}
		msgs, err := c.encode(ev)
		data, _ := json.Marshal(msgs)
		return string(data), err
	}
	image := &provider.RealtimeMedia{MIMEType: "image/png", Data: []byte{1}}
	user := provider.Message{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}

	for _, c := range []struct {
		delegation string
		ev         provider.RealtimeClientEvent
		want       string
	}{
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeTextInput, Text: "hi"},
			`[{"item":{"content":[{"text":"hi","type":"input_text"}],"role":"user","type":"message"},"type":"response.item.create"},{"type":"response.create"}]`},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeImageInput, Media: image},
			`[{"item":{"content":[{"image_url":"data:image/png;base64,AQ==","type":"input_image"}],"role":"user","type":"message"},"type":"response.item.create"}]`},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeConversationAppend, Conversation: &provider.RealtimeConversation{Messages: []provider.Message{user}, RequestResponse: true}},
			`[{"item":{"content":[{"text":"hi","type":"input_text"}],"role":"user","type":"message"},"type":"response.item.create"},{"type":"response.create"}]`},
		{"decisions", provider.RealtimeClientEvent{Type: provider.RealtimeTextInput, Text: "hi"},
			`[{"item":{"content":[{"text":"hi","type":"input_text"}],"role":"user","type":"message"},"type":"decision.item.create"},{"type":"decision.create"}]`},
		{"decisions", provider.RealtimeClientEvent{Type: provider.RealtimeImageInput, Media: image},
			`[{"item":{"content":[{"image_url":"data:image/png;base64,AQ==","type":"input_image"}],"role":"user","type":"message"},"type":"decision.item.create"}]`},
		{"decisions", provider.RealtimeClientEvent{Type: provider.RealtimeResponseCreate}, `[{"type":"decision.create"}]`},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: &provider.RealtimeSessionUpdate{
			Tools: []provider.ToolDefinition{}, ProviderOptions: map[string]any{"triggers": []any{}},
		}}, `[{"session":{"delegation":{"responses":{"tools":[]},"type":"responses"},"triggers":[]},"type":"session.update"}]`},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{
			ToolCallID: "c1", Output: json.RawMessage(`"ok"`), ProviderOptions: map[string]any{"status": "completed"},
		}}, `[{"item":{"call_id":"c1","output":"\"ok\"","status":"completed","type":"function_call_output"},"type":"response.item.create"}]`},
	} {
		got, err := encode(c.delegation, c.ev)
		if err != nil || got != c.want {
			t.Errorf("%s %s:\ngot  %s (%v)\nwant %s", c.delegation, c.ev.Type, got, err, c.want)
		}
	}

	assistant := provider.Message{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartText, Text: "x"}}}
	system := "x"
	for _, c := range []struct {
		delegation string
		ev         provider.RealtimeClientEvent
		want       error
	}{
		{"client", provider.RealtimeClientEvent{Type: provider.RealtimeTextInput, Text: "hi"}, provider.ErrRealtimeUnsupported},
		{"client", provider.RealtimeClientEvent{Type: provider.RealtimeImageInput, Media: image}, provider.ErrRealtimeUnsupported},
		{"client", provider.RealtimeClientEvent{Type: provider.RealtimeResponseCreate}, provider.ErrRealtimeUnsupported},
		{"client", provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: &provider.RealtimeSessionUpdate{Tools: []provider.ToolDefinition{}}}, provider.ErrRealtimeUnsupported},
		{"client", provider.RealtimeClientEvent{Type: provider.RealtimeConversationAppend, Conversation: &provider.RealtimeConversation{Messages: []provider.Message{user}}}, provider.ErrRealtimeUnsupported},
		{"decisions", provider.RealtimeClientEvent{Type: provider.RealtimeConversationAppend, Conversation: &provider.RealtimeConversation{Messages: []provider.Message{assistant}}}, provider.ErrInvalidRealtimeConfig},
		{"decisions", provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{ToolCallID: "c"}}, provider.ErrRealtimeUnsupported},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeResponseCreate, ProviderOptions: map[string]any{"x": 1}}, provider.ErrRealtimeUnsupported},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeImageInput, Media: &provider.RealtimeMedia{MIMEType: "image/png"}}, provider.ErrInvalidRealtimeConfig},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeConversationAppend}, provider.ErrInvalidRealtimeConfig},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeConversationAppend, Conversation: &provider.RealtimeConversation{Messages: []provider.Message{{Role: provider.RoleUser}}}}, provider.ErrInvalidRealtimeConfig},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: &provider.RealtimeSessionUpdate{System: &system}}, provider.ErrRealtimeUnsupported},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent}, provider.ErrInvalidRealtimeConfig},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: &provider.RealtimeSessionUpdate{}}, provider.ErrInvalidRealtimeConfig},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: &provider.RealtimeSessionUpdate{ProviderOptions: map[string]any{"model": "x"}}}, provider.ErrInvalidRealtimeConfig},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: &provider.RealtimeSessionUpdate{Tools: []provider.ToolDefinition{{Name: "t", InputSchema: json.RawMessage(`{bad`)}}}}, provider.ErrInvalidRealtimeConfig},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{ToolCallID: "c", Partial: true}}, provider.ErrRealtimeUnsupported},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{ToolCallID: "c", ProviderOptions: map[string]any{"call_id": "x"}}}, provider.ErrInvalidRealtimeConfig},
		{"responses", provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioEnd}, provider.ErrRealtimeUnsupported},
	} {
		if _, err := encode(c.delegation, c.ev); !errors.Is(err, c.want) {
			t.Errorf("%s %s: err = %v, want %v", c.delegation, c.ev.Type, err, c.want)
		}
	}
}

func TestLiveTranscriptTimingAndUsageMetadata(t *testing.T) {
	c := &liveConn{pending: map[string]bool{}, inputTr: true, outputTr: true}
	for _, typ := range []string{"session.input_transcript.delta", "session.output_transcript.delta"} {
		evs, _ := c.translate(liveServerEvent{Type: typ, Delta: "x", StartMs: 100, EndMs: 900})
		if evs[0].AudioStart != 100*time.Millisecond || evs[0].AudioEnd != 900*time.Millisecond {
			t.Errorf("%s timing = %+v", typ, evs[0])
		}
	}
	var ev liveServerEvent
	_ = json.Unmarshal([]byte(`{"type":"session.closed","reason":"client","usage":{"seconds":2},"context_window":{"usage_ratio":0.25}}`), &ev)
	evs, _ := c.translate(ev)
	meta, _ := evs[0].ProviderMetadata["openai"].(map[string]any)
	if meta["contextWindowUsageRatio"] != 0.25 || meta["closeReason"] != "client" {
		t.Errorf("usage metadata = %+v", evs[0].ProviderMetadata)
	}
}

func TestLiveSessionContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/live/sessions/sess_1/content" && r.Header.Get("Authorization") == "Bearer k" && r.Header.Get("X-Team") == "voice":
			w.Header().Set("Content-Type", "audio/wav")
			_, _ = w.Write([]byte("RIFF"))
		default:
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"message":"no such session"}}`))
		}
	}))
	defer server.Close()

	opts := []Option{WithAPIKey("k"), WithBaseURL(server.URL + "/v1"), WithHeaders(map[string]string{"X-Team": "voice"})}
	body, contentType, err := LiveSessionContent(t.Context(), "sess_1", opts...)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(body)
	_ = body.Close()
	if string(data) != "RIFF" || contentType != "audio/wav" {
		t.Fatalf("content = %q, %q", data, contentType)
	}
	if _, _, err := LiveSessionContent(t.Context(), "missing", opts...); err == nil || !strings.Contains(err.Error(), "no such session") {
		t.Errorf("missing session: %v", err)
	}
	if _, _, err := LiveSessionContent(t.Context(), ""); err == nil {
		t.Error("an empty session ID must fail")
	}
	t.Setenv("OPENAI_API_KEY", "")
	if _, _, err := LiveSessionContent(t.Context(), "sess_1", WithBaseURL(server.URL)); err == nil {
		t.Error("missing credentials must fail")
	}
}
