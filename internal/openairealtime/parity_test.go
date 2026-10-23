package openairealtime

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
)

func mustNext(t *testing.T, s *sequencer) clientEvent {
	t.Helper()
	ev, ok := s.next()
	if !ok {
		t.Fatal("expected a response.create")
	}
	return ev
}

func expectIdle(t *testing.T, s *sequencer) {
	t.Helper()
	if ev, ok := s.next(); ok {
		t.Fatalf("unexpected response.create %v", ev)
	}
}

func TestSequencerOrdersRequests(t *testing.T) {
	s := newSequencer()
	s.created("r0", true) // a response is active: everything waits

	formal := map[string]any{"instructions": "formal"}
	brief := map[string]any{"instructions": "brief"}
	s.request(nil)
	s.request(nil) // coalesces with the previous automatic request
	s.request(formal)
	s.request(brief) // requests with parameters never coalesce
	s.request(nil)
	expectIdle(t, s)

	var sent []any
	active := "r0"
	for i := range 4 {
		s.done(active)
		ev := mustNext(t, s)
		expectIdle(t, s) // an unconfirmed create holds the slot
		sent = append(sent, ev["response"])
		active = fmt.Sprintf("r%d", i+1)
		s.created(active, true)
	}
	s.done(active)
	expectIdle(t, s)
	want := []any{nil, formal, brief, nil}
	if !reflect.DeepEqual(sent, want) {
		t.Fatalf("sent %v, want %v", sent, want)
	}
}

func TestSequencerEventIDsAndRejections(t *testing.T) {
	s := newSequencer()
	s.request(nil)
	first := mustNext(t, s)
	if first["event_id"] != "goai_response_create_1" {
		t.Fatalf("event_id = %v", first["event_id"])
	}

	// An error about another event leaves the request in flight.
	if s.rejected(&wireError{EventID: "other"}) || s.rejected(&wireError{Code: "unrelated"}) || s.rejected(nil) {
		t.Fatal("unrelated errors must not drop the request")
	}
	if !s.rejected(&wireError{EventID: "goai_response_create_1", Code: "conversation_already_has_active_response"}) {
		t.Fatal("the rejected create must be dropped")
	}
	expectIdle(t, s) // dropped, not retried

	s.request(nil)
	mustNext(t, s)
	if !s.rejected(&wireError{Code: "bad_response_create"}) {
		t.Fatal("a create error without event_id must match")
	}
	s.request(nil)
	mustNext(t, s)
	if !s.rejected(&wireError{Message: "Invalid response.create payload"}) {
		t.Fatal("a create error named in the message must match")
	}

	s.request(nil)
	ev := mustNext(t, s)
	s.sendFailed("not-it")
	expectIdle(t, s)
	s.sendFailed(ev["event_id"].(string))
	s.request(nil)
	mustNext(t, s) // the slot is free again
}

func TestSequencerActiveResponseAndOutOfBand(t *testing.T) {
	s := newSequencer()
	s.created("main", true)
	s.done("other") // the end of another response keeps "main" active
	if !s.active() {
		t.Fatal("response.done of another response freed the active one")
	}

	// An out-of-band response does not reset the conversation's tool turn.
	s.toolCall("c1", "main")
	s.done("main")
	s.created("oob", false)
	s.toolCall("c-oob", "oob") // its calls do not hold the conversation back
	s.done("oob")
	s.toolOutput("c1")
	if ev := mustNext(t, s); ev["response"] != nil {
		t.Fatalf("tool continuation must be automatic: %v", ev)
	}
	s.done("") // a response.done without ID ends whatever is active
	if s.active() {
		t.Fatal("still active")
	}
}

func TestEncodeParityCommands(t *testing.T) {
	c := newTestConn()
	c.autoResponse = true
	encode := func(ev provider.RealtimeClientEvent) ([]clientEvent, *createRequest, error) {
		t.Helper()
		return c.encode(ev)
	}
	asJSON := func(v any) string {
		data, _ := json.Marshal(v)
		return string(data)
	}

	msgs, resp, err := encode(provider.RealtimeClientEvent{Type: provider.RealtimeImageInput, Media: &provider.RealtimeMedia{MIMEType: "image/png", Data: []byte{1}}})
	if err != nil || resp != nil || asJSON(msgs) != `[{"item":{"content":[{"image_url":"data:image/png;base64,AQ==","type":"input_image"}],"role":"user","type":"message"},"type":"conversation.item.create"}]` {
		t.Fatalf("image = %s, %v, %v", asJSON(msgs), resp, err)
	}
	if _, _, err := encode(provider.RealtimeClientEvent{Type: provider.RealtimeImageInput, Media: &provider.RealtimeMedia{MIMEType: "image/png"}}); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
		t.Fatalf("empty image: %v", err)
	}
	if _, _, err := encode(provider.RealtimeClientEvent{Type: provider.RealtimeImageInput, Media: &provider.RealtimeMedia{MIMEType: "audio/wav", Data: []byte{1}}}); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
		t.Fatalf("non-image media: %v", err)
	}

	conv := &provider.RealtimeConversation{RequestResponse: true, Messages: []provider.Message{
		{Role: provider.RoleSystem, Content: []provider.Part{{Type: provider.PartText, Text: "Be kind."}}},
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "Hi"}, {Type: provider.PartImage, URL: "data:image/png;base64,AQ==", Detail: "low"}}},
		{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartText, Text: "Checking."}, {Type: provider.PartToolCall, ToolCallID: "c1", ToolName: "look"}}},
		{Role: provider.RoleTool, Content: []provider.Part{{Type: provider.PartToolResult, ToolCallID: "c1", ToolOutput: "done"}}},
	}}
	msgs, resp, err = encode(provider.RealtimeClientEvent{Type: provider.RealtimeConversationAppend, Conversation: conv})
	if err != nil || resp == nil || resp.params != nil {
		t.Fatalf("conversation: %v %v", resp, err)
	}
	want := `[{"item":{"content":[{"text":"Be kind.","type":"input_text"}],"role":"system","type":"message"},"type":"conversation.item.create"},` +
		`{"item":{"content":[{"text":"Hi","type":"input_text"},{"detail":"low","image_url":"data:image/png;base64,AQ==","type":"input_image"}],"role":"user","type":"message"},"type":"conversation.item.create"},` +
		`{"item":{"content":[{"text":"Checking.","type":"output_text"}],"role":"assistant","type":"message"},"type":"conversation.item.create"},` +
		`{"item":{"arguments":"{}","call_id":"c1","name":"look","type":"function_call"},"type":"conversation.item.create"},` +
		`{"item":{"call_id":"c1","output":"done","type":"function_call_output"},"type":"conversation.item.create"}]`
	if got := asJSON(msgs); got != want {
		t.Fatalf("conversation items\ngot  %s\nwant %s", got, want)
	}
	conv.RequestResponse = false
	if _, resp, _ := encode(provider.RealtimeClientEvent{Type: provider.RealtimeConversationAppend, Conversation: conv}); resp != nil {
		t.Fatal("history without RequestResponse must not ask for a response")
	}
	for _, bad := range []*provider.RealtimeConversation{
		nil, {},
		{Messages: []provider.Message{{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartImage, URL: "x"}}}}},
		{Messages: []provider.Message{{Role: provider.RoleUser}}},
	} {
		if _, _, err := encode(provider.RealtimeClientEvent{Type: provider.RealtimeConversationAppend, Conversation: bad}); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
			t.Errorf("conversation %+v: %v", bad, err)
		}
	}

	system := "Be brief."
	msgs, _, err = encode(provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: &provider.RealtimeSessionUpdate{
		System: &system, Tools: []provider.ToolDefinition{}, ProviderOptions: map[string]any{"audio": map[string]any{"output": map[string]any{"speed": 1.2}}},
	}})
	if err != nil || asJSON(msgs) != `[{"session":{"audio":{"output":{"speed":1.2}},"instructions":"Be brief.","tools":[],"type":"realtime"},"type":"session.update"}]` {
		t.Fatalf("session update = %s, %v", asJSON(msgs), err)
	}
	for _, bad := range []*provider.RealtimeSessionUpdate{nil, {}, {ProviderOptions: map[string]any{"instructions": "x"}}} {
		if _, _, err := encode(provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: bad}); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
			t.Errorf("update %+v: %v", bad, err)
		}
	}

	params := map[string]any{"conversation": "none", "instructions": "classify"}
	if _, resp, err := encode(provider.RealtimeClientEvent{Type: provider.RealtimeResponseCreate, ProviderOptions: params}); err != nil || !reflect.DeepEqual(resp.params, params) {
		t.Fatalf("response with parameters = %v, %v", resp, err)
	}

	msgs, _, err = encode(provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{
		ToolCallID: "c1", Output: json.RawMessage(`{}`), ProviderOptions: map[string]any{"id": "item_1"},
	}})
	if err != nil || !strings.Contains(asJSON(msgs), `"id":"item_1"`) {
		t.Fatalf("tool output options = %s, %v", asJSON(msgs), err)
	}
	for _, r := range []*provider.RealtimeToolResult{
		{ToolCallID: "c1", Partial: true},
		{ToolCallID: "c1", Media: []provider.RealtimeMedia{{MIMEType: "image/png", Data: []byte{1}}}},
	} {
		if _, _, err := encode(provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: r}); !errors.Is(err, provider.ErrRealtimeUnsupported) {
			t.Errorf("tool result %+v: %v", r, err)
		}
	}
	if _, _, err := encode(provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{ToolCallID: "c1", ProviderOptions: map[string]any{"call_id": "x"}}}); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
		t.Errorf("overriding call_id: %v", err)
	}
	if _, _, err := encode(provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioEnd}); !errors.Is(err, provider.ErrRealtimeUnsupported) {
		t.Errorf("audio end: %v", err)
	}
}

func TestTranslateParityEvents(t *testing.T) {
	c := newTestConn()
	decode := func(raw string) []provider.RealtimeEvent {
		t.Helper()
		var ev serverEvent
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			t.Fatal(err)
		}
		events, known := c.translate(ev)
		if !known {
			t.Fatalf("%s not known", raw)
		}
		return events
	}
	openai := func(ev provider.RealtimeEvent) map[string]any {
		m, _ := ev.ProviderMetadata["openai"].(map[string]any)
		return m
	}

	if ev := decode(`{"type":"input_audio_buffer.speech_started","item_id":"u","audio_start_ms":120}`)[0]; ev.AudioStart != 120*time.Millisecond {
		t.Errorf("speech started = %+v", ev)
	}
	if ev := decode(`{"type":"input_audio_buffer.speech_stopped","item_id":"u","audio_end_ms":900}`)[0]; ev.AudioEnd != 900*time.Millisecond {
		t.Errorf("speech stopped = %+v", ev)
	}
	if ev := decode(`{"type":"conversation.item.input_audio_transcription.delta","item_id":"u","delta":"ho","logprobs":[{"token":"ho","logprob":-0.1}]}`)[0]; openai(ev)["logprobs"] == nil {
		t.Errorf("delta logprobs = %+v", ev)
	}
	ev := decode(`{"type":"conversation.item.input_audio_transcription.completed","item_id":"u","transcript":"hola","usage":{"type":"duration","seconds":2}}`)[0]
	if !ev.IsFinal || openai(ev)["usage"] == nil || openai(ev)["logprobs"] != nil {
		t.Errorf("completed = %+v", ev)
	}
	ev = decode(`{"type":"conversation.item.input_audio_transcription.segment","item_id":"u","id":"seg_1","text":"hola","speaker":"A","start":0.5,"end":1.25}`)[0]
	if ev.Speaker != "A" || ev.AudioStart != 500*time.Millisecond || ev.AudioEnd != 1250*time.Millisecond || ev.IsFinal || openai(ev)["segmentId"] != "seg_1" {
		t.Errorf("segment = %+v", ev)
	}
	if ev := decode(`{"type":"response.output_text.done","response_id":"r","item_id":"t","text":"full text"}`)[0]; ev.Type != provider.RealtimeTextDelta || !ev.IsFinal || ev.Text != "full text" {
		t.Errorf("text done = %+v", ev)
	}

	ev = decode(`{"type":"response.created","response":{"id":"oob_1","conversation_id":null,"metadata":{"purpose":"classify"}}}`)[0]
	if ev.ResponseID != "oob_1" || openai(ev)["metadata"] == nil || openai(ev)["conversationId"] != nil {
		t.Errorf("out-of-band created = %+v", ev)
	}
	if !c.responses.outOfBand["oob_1"] {
		t.Error("a null conversation_id marks the response out of band")
	}
	events := decode(`{"type":"response.done","response":{"id":"oob_1","status":"completed","conversation_id":null,"metadata":{"purpose":"classify"},
		"usage":{"total_tokens":10,"input_tokens":6,"output_tokens":4,"input_token_details":{"cached_tokens":2,"audio_tokens":3},"output_token_details":{"text_tokens":4}}}}`)
	if len(events) != 2 || openai(events[0])["metadata"] == nil {
		t.Fatalf("done = %+v", events)
	}
	if u := events[1]; u.Usage.CacheReadTokens != 2 || u.Usage.InputTokens != 4 || openai(u)["inputTokenDetails"] == nil || openai(u)["outputTokenDetails"] == nil {
		t.Errorf("usage = %+v", u)
	}
	if c.metadata("x", json.RawMessage(`null`), "y", "", "z", map[string]any{}) != nil {
		t.Error("empty values must leave metadata nil")
	}
}

func TestRequestResponseWithParameters(t *testing.T) {
	verifyNoLeaks(t)

	server, received, _ := scriptedServer(t)
	conn, err := testModel(server.URL).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()
	next(t, conn) // ready

	err = conn.Send(t.Context(), provider.RealtimeClientEvent{
		Type:            provider.RealtimeResponseCreate,
		ProviderOptions: map[string]any{"conversation": "none", "metadata": map[string]any{"purpose": "classify"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	expectMessages(t, received, `{"response":{"conversation":"none","metadata":{"purpose":"classify"}},"type":"response.create"}`)
}
