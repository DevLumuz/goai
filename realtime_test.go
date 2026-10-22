package goai

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
	"go.uber.org/goleak"
)

type fakeRealtimeConnection struct {
	events chan provider.RealtimeEvent
	sent   chan provider.RealtimeClientEvent

	closeOnce sync.Once
	closed    chan struct{}

	errMu sync.RWMutex
	err   error
}

func newFakeRealtimeConnection(eventBuffer, sentBuffer int) *fakeRealtimeConnection {
	return &fakeRealtimeConnection{
		events: make(chan provider.RealtimeEvent, eventBuffer),
		sent:   make(chan provider.RealtimeClientEvent, sentBuffer),
		closed: make(chan struct{}),
	}
}

func (c *fakeRealtimeConnection) Events() <-chan provider.RealtimeEvent { return c.events }

func (c *fakeRealtimeConnection) Send(ctx context.Context, event provider.RealtimeClientEvent) error {
	select {
	case <-c.closed:
		return provider.ErrRealtimeSessionClosed
	default:
	}
	select {
	case c.sent <- event:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-c.closed:
		return provider.ErrRealtimeSessionClosed
	}
}

func (c *fakeRealtimeConnection) Close() error {
	c.closeOnce.Do(func() { close(c.closed) })
	return nil
}

func (c *fakeRealtimeConnection) Err() error {
	c.errMu.RLock()
	defer c.errMu.RUnlock()
	return c.err
}

func (c *fakeRealtimeConnection) finish(err error) {
	c.errMu.Lock()
	c.err = err
	c.errMu.Unlock()
	close(c.events)
}

type fakeRealtimeModel struct {
	id     string
	conn   provider.RealtimeConnection
	err    error
	openFn func(context.Context, provider.RealtimeParams) (provider.RealtimeConnection, error)
}

func (m *fakeRealtimeModel) ModelID() string { return m.id }
func (m *fakeRealtimeModel) OpenRealtime(ctx context.Context, params provider.RealtimeParams) (provider.RealtimeConnection, error) {
	if m.openFn != nil {
		return m.openFn(ctx, params)
	}
	return m.conn, m.err
}

func receiveRealtimeEvent(t *testing.T, events <-chan provider.RealtimeEvent) provider.RealtimeEvent {
	t.Helper()
	select {
	case event, ok := <-events:
		if !ok {
			t.Fatal("realtime event channel closed unexpectedly")
		}
		return event
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for realtime event")
		return provider.RealtimeEvent{}
	}
}

func receiveRealtimeClientEvent(t *testing.T, events <-chan provider.RealtimeClientEvent) provider.RealtimeClientEvent {
	t.Helper()
	select {
	case event := <-events:
		return event
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for realtime client event")
		return provider.RealtimeClientEvent{}
	}
}

func waitRealtimeDone(t *testing.T, session *RealtimeSession) {
	t.Helper()
	select {
	case <-session.done:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for realtime session to close")
	}
}

func TestOpenRealtimeSessionValidation(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(1, 1)
	model := &fakeRealtimeModel{id: "test", conn: conn}

	var nilCtx context.Context
	if _, err := OpenRealtimeSession(nilCtx, model); err == nil {
		t.Fatal("expected nil context error")
	}
	if _, err := OpenRealtimeSession(t.Context(), nil); err == nil {
		t.Fatal("expected nil model error")
	}
	if _, err := OpenRealtimeSession(t.Context(), model, WithRealtimeEventBuffer(0)); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
		t.Fatalf("event buffer error = %v", err)
	}
	if _, err := OpenRealtimeSession(t.Context(), model, WithRealtimeBackpressureTimeout(-time.Second)); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
		t.Fatalf("backpressure timeout error = %v", err)
	}
	if _, err := OpenRealtimeSession(t.Context(), model, WithRealtimeMaxParallelTools(0)); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
		t.Fatalf("parallel tools error = %v", err)
	}
	if _, err := OpenRealtimeSession(t.Context(), model, WithRealtimeMaxOutputTokens(-1)); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
		t.Fatalf("params error = %v", err)
	}

	openErr := errors.New("open failed")
	if _, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{err: openErr}); !errors.Is(err, openErr) {
		t.Fatalf("open error = %v", err)
	}
	if _, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{}); err == nil {
		t.Fatal("expected nil connection error")
	}
}

func TestRealtimeSessionOptionsAreCopied(t *testing.T) {
	defer goleak.VerifyNone(t)

	providerOptions := map[string]any{
		"nested": map[string]any{"value": "original"},
		"bytes":  []byte{1, 2, 3},
	}
	toolSchema := json.RawMessage(`{"type":"object"}`)
	captured := make(chan provider.RealtimeParams, 1)
	conn := newFakeRealtimeConnection(1, 1)
	model := &fakeRealtimeModel{id: "test", openFn: func(_ context.Context, params provider.RealtimeParams) (provider.RealtimeConnection, error) {
		captured <- params
		return conn, nil
	}}

	session, err := OpenRealtimeSession(t.Context(), model,
		WithRealtimeSystem("system"),
		WithRealtimeAudio(
			provider.AudioFormat{Encoding: provider.AudioPCM16, SampleRate: 16000, Channels: 1},
			provider.AudioFormat{Encoding: provider.AudioPCM16, SampleRate: 24000, Channels: 1},
		),
		WithRealtimeTools(Tool{Name: "report", Description: "report", InputSchema: toolSchema}),
		WithRealtimeProviderOptions(providerOptions),
		WithRealtimeToolChoice("report"),
		WithRealtimeVoice("marin"),
		WithRealtimeTurnDetection(provider.TurnDetection{Mode: provider.TurnModeClientVAD, ProviderOptions: map[string]any{"k": []any{"v"}}}),
		WithRealtimeInputTranscription(provider.TranscriptionConfig{Enabled: true, Language: "es"}),
		WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true}),
		WithRealtimeSequentialTools(true),
		nil, // nil options are ignored
	)
	if err != nil {
		t.Fatal(err)
	}

	providerOptions["nested"].(map[string]any)["value"] = "changed"
	providerOptions["bytes"].([]byte)[0] = 9
	toolSchema[0] = 'x'

	params := <-captured
	if got := params.ProviderOptions["nested"].(map[string]any)["value"]; got != "original" {
		t.Fatalf("nested option mutated: %v", got)
	}
	if got := params.ProviderOptions["bytes"].([]byte)[0]; got != 1 {
		t.Fatalf("byte option mutated: %v", got)
	}
	if got := string(params.Tools[0].InputSchema); got != `{"type":"object"}` {
		t.Fatalf("tool schema mutated: %s", got)
	}
	if params.ToolChoice != "report" || params.Voice != "marin" || params.TurnDetection.Mode != provider.TurnModeClientVAD ||
		!params.InputTranscription.Enabled || params.InputTranscription.Language != "es" || !params.OutputTranscription.Enabled {
		t.Fatalf("options not applied: %+v", params)
	}
	if cap(session.toolSlots) != 1 {
		t.Fatalf("sequential tools must allow one slot, got %d", cap(session.toolSlots))
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRealtimeSessionInputMethodsAndCopies(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(1, 16)
	session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn})
	if err != nil {
		t.Fatal(err)
	}

	audio := []byte{1, 2, 3}
	if err := session.InputAudio(t.Context(), audio); err != nil {
		t.Fatal(err)
	}
	audio[0] = 9
	got := receiveRealtimeClientEvent(t, conn.sent)
	if got.Type != provider.RealtimeInputAudioAppend || got.Audio[0] != 1 {
		t.Fatalf("audio event = %+v", got)
	}

	calls := []struct {
		invoke func() error
		want   provider.RealtimeClientEventType
	}{
		{func() error { return session.CommitInput(t.Context()) }, provider.RealtimeInputAudioCommit},
		{func() error { return session.StartInputActivity(t.Context()) }, provider.RealtimeInputActivityStart},
		{func() error { return session.EndInputActivity(t.Context()) }, provider.RealtimeInputActivityEnd},
		{func() error { return session.SendText(t.Context(), "hello") }, provider.RealtimeTextInput},
		{func() error { return session.RequestResponse(t.Context()) }, provider.RealtimeResponseCreate},
		{func() error { return session.RequestResponseWith(t.Context(), map[string]any{"k": "v"}) }, provider.RealtimeResponseCreate},
	}
	for _, call := range calls {
		if err := call.invoke(); err != nil {
			t.Fatal(err)
		}
		if event := receiveRealtimeClientEvent(t, conn.sent); event.Type != call.want {
			t.Fatalf("event type = %q, want %q", event.Type, call.want)
		}
	}

	content := provider.RealtimeContext{Kind: provider.RealtimeContextCommentary, DelegationID: "d1", Text: "done"}
	if err := session.AppendContext(t.Context(), content); err != nil {
		t.Fatal(err)
	}
	if got := receiveRealtimeClientEvent(t, conn.sent); got.Type != provider.RealtimeContextAppend || got.Context == nil || *got.Context != content {
		t.Fatalf("context event = %+v", got)
	}

	cursor := &provider.PlaybackCursor{ResponseID: "r", ItemID: "i", PlayedFor: 250 * time.Millisecond}
	if err := session.Interrupt(t.Context(), cursor); err != nil {
		t.Fatal(err)
	}
	cursor.ItemID = "changed"
	interrupt := receiveRealtimeClientEvent(t, conn.sent)
	if interrupt.Cursor == nil || interrupt.Cursor.ItemID != "i" {
		t.Fatalf("interrupt cursor = %+v", interrupt.Cursor)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if err := session.SendText(t.Context(), "late"); !errors.Is(err, provider.ErrRealtimeSessionClosed) {
		t.Fatalf("send after close = %v", err)
	}
}

func TestRealtimeSessionPublishesDefensiveEventCopyAndHooks(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(2, 1)
	hookEvents := make(chan provider.RealtimeEvent, 1)
	var panics atomic.Int32
	session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn},
		WithRealtimeOnEvent(func(provider.RealtimeEvent) { panic("hook panic") }),
		WithRealtimeOnEvent(func(event provider.RealtimeEvent) { hookEvents <- event }),
		WithRealtimeOnPanic(func(PanicInfo) { panics.Add(1) }),
	)
	if err != nil {
		t.Fatal(err)
	}

	audio := []byte{1, 2}
	metadata := map[string]any{"nested": map[string]any{"value": "original"}}
	conn.events <- provider.RealtimeEvent{
		Type:             provider.RealtimeOutputAudio,
		Audio:            audio,
		ProviderMetadata: metadata,
	}

	published := receiveRealtimeEvent(t, session.Events())
	hooked := receiveRealtimeEvent(t, hookEvents)
	audio[0] = 9
	metadata["nested"].(map[string]any)["value"] = "changed"
	if published.Audio[0] != 1 || hooked.Audio[0] != 1 {
		t.Fatalf("audio was not copied: published=%v hooked=%v", published.Audio, hooked.Audio)
	}
	if got := published.ProviderMetadata["nested"].(map[string]any)["value"]; got != "original" {
		t.Fatalf("metadata was not copied: %v", got)
	}
	if panics.Load() != 1 {
		t.Fatalf("panic callbacks = %d", panics.Load())
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRealtimeSessionConnectionErrorAndNormalFinish(t *testing.T) {
	defer goleak.VerifyNone(t)

	t.Run("terminal error", func(t *testing.T) {
		conn := newFakeRealtimeConnection(1, 1)
		session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn})
		if err != nil {
			t.Fatal(err)
		}
		terminal := errors.New("connection failed")
		conn.finish(terminal)
		waitRealtimeDone(t, session)
		if !errors.Is(session.Err(), terminal) {
			t.Fatalf("session error = %v", session.Err())
		}
		if err := session.Close(); !errors.Is(err, terminal) {
			t.Fatalf("Close error = %v", err)
		}
	})

	t.Run("normal finish", func(t *testing.T) {
		conn := newFakeRealtimeConnection(1, 1)
		session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn})
		if err != nil {
			t.Fatal(err)
		}
		conn.finish(nil)
		waitRealtimeDone(t, session)
		if session.Err() != nil {
			t.Fatalf("session error = %v", session.Err())
		}
		if err := session.Close(); err != nil {
			t.Fatal(err)
		}
	})
}

func TestRealtimeSessionCloseIsConcurrentAndIdempotent(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(1, 1)
	session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn})
	if err != nil {
		t.Fatal(err)
	}

	const callers = 10
	errs := make(chan error, callers)
	var wg sync.WaitGroup
	wg.Add(callers)
	for range callers {
		go func() {
			defer wg.Done()
			errs <- session.Close()
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("Close error = %v", err)
		}
	}
}

func TestRealtimeSessionBackpressure(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(4, 1)
	session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn},
		WithRealtimeEventBuffer(1), WithRealtimeBackpressureTimeout(10*time.Millisecond))
	if err != nil {
		t.Fatal(err)
	}
	conn.events <- provider.RealtimeEvent{Type: provider.RealtimeTextDelta, Text: "one"}
	conn.events <- provider.RealtimeEvent{Type: provider.RealtimeTextDelta, Text: "two"}
	waitRealtimeDone(t, session)
	if !errors.Is(session.Err(), provider.ErrRealtimeBackpressure) {
		t.Fatalf("session error = %v", session.Err())
	}
	if err := session.Close(); !errors.Is(err, provider.ErrRealtimeBackpressure) {
		t.Fatalf("Close error = %v", err)
	}
}

func TestRealtimeSessionSurvivesTransientBackpressure(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(4, 1)
	session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn},
		WithRealtimeEventBuffer(1), WithRealtimeBackpressureTimeout(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	// The second event finds the queue full; draining it later must not fail the session.
	conn.events <- provider.RealtimeEvent{Type: provider.RealtimeTextDelta, Text: "one"}
	conn.events <- provider.RealtimeEvent{Type: provider.RealtimeTextDelta, Text: "two"}
	for _, want := range []string{"one", "two"} {
		if got := receiveRealtimeEvent(t, session.Events()); got.Text != want {
			t.Fatalf("event text = %q, want %q", got.Text, want)
		}
	}
	if err := session.Err(); err != nil {
		t.Fatalf("session error = %v", err)
	}
}

func TestRealtimeSessionDispatchesToolResult(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(4, 4)
	var starts atomic.Int32
	var finishes atomic.Int32
	tool := NewTool("report", "Create report", func(_ context.Context, input struct {
		Month string `json:"month"`
	}) (string, error) {
		return `{"report":"` + input.Month + `"}`, nil
	})
	session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn},
		WithRealtimeTools(tool),
		WithRealtimeOnToolCallStart(func(info ToolCallStartInfo) {
			if info.ToolCallID != "call-1" || info.ToolName != "report" {
				t.Errorf("unexpected start info: %+v", info)
			}
			starts.Add(1)
		}),
		WithRealtimeOnToolCall(func(info ToolCallInfo) {
			if info.Error != nil || info.Output == "" {
				t.Errorf("unexpected tool info: %+v", info)
			}
			finishes.Add(1)
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	conn.events <- provider.RealtimeEvent{
		Type: provider.RealtimeToolCallEvent,
		ToolCall: &provider.RealtimeToolCall{
			ID: "call-1", Name: "report", Input: json.RawMessage(`{"month":"last"}`),
		},
	}
	_ = receiveRealtimeEvent(t, session.Events())
	resultEvent := receiveRealtimeClientEvent(t, conn.sent)
	if resultEvent.Type != provider.RealtimeToolResultEvent || resultEvent.ToolResult == nil {
		t.Fatalf("tool result event = %+v", resultEvent)
	}
	if resultEvent.ToolResult.ToolCallID != "call-1" || resultEvent.ToolResult.IsError {
		t.Fatalf("tool result = %+v", resultEvent.ToolResult)
	}
	if got := string(resultEvent.ToolResult.Output); got != `{"report":"last"}` {
		t.Fatalf("tool output = %s", got)
	}
	if starts.Load() != 1 || finishes.Load() != 1 {
		t.Fatalf("hooks starts=%d finishes=%d", starts.Load(), finishes.Load())
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRealtimeSessionToolErrorsAndUnknownTool(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(8, 8)
	tool := Tool{
		Name: "fail",
		Execute: func(context.Context, json.RawMessage) (string, error) {
			return "", errors.New("failed")
		},
	}
	session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn}, WithRealtimeTools(tool))
	if err != nil {
		t.Fatal(err)
	}

	for _, call := range []provider.RealtimeToolCall{
		{ID: "known", Name: "fail", Input: json.RawMessage(`{}`)},
		{ID: "unknown", Name: "missing", Input: json.RawMessage(`{}`)},
	} {
		copy := call
		conn.events <- provider.RealtimeEvent{Type: provider.RealtimeToolCallEvent, ToolCall: &copy}
		_ = receiveRealtimeEvent(t, session.Events())
	}

	results := map[string]*provider.RealtimeToolResult{}
	for range 2 {
		event := receiveRealtimeClientEvent(t, conn.sent)
		results[event.ToolResult.ToolCallID] = event.ToolResult
	}
	for _, id := range []string{"known", "unknown"} {
		result := results[id]
		if result == nil || !result.IsError || !json.Valid(result.Output) {
			t.Fatalf("result %q = %+v", id, result)
		}
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRealtimeSessionCancelsToolCall(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(8, 2)
	started := make(chan struct{})
	cancelled := make(chan struct{})
	tool := Tool{
		Name: "slow",
		Execute: func(ctx context.Context, _ json.RawMessage) (string, error) {
			close(started)
			<-ctx.Done()
			close(cancelled)
			return "", ctx.Err()
		},
	}
	session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn}, WithRealtimeTools(tool))
	if err != nil {
		t.Fatal(err)
	}

	conn.events <- provider.RealtimeEvent{
		Type: provider.RealtimeToolCallEvent,
		ToolCall: &provider.RealtimeToolCall{
			ID: "call-cancel", Name: "slow", Input: json.RawMessage(`{}`),
		},
	}
	_ = receiveRealtimeEvent(t, session.Events())
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("tool did not start")
	}
	conn.events <- provider.RealtimeEvent{Type: provider.RealtimeToolCallCancelled, ToolCallID: "call-cancel"}
	_ = receiveRealtimeEvent(t, session.Events())
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("tool was not cancelled")
	}

	select {
	case event := <-conn.sent:
		t.Fatalf("cancelled tool sent stale result: %+v", event)
	case <-time.After(50 * time.Millisecond):
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRealtimeSessionToolTimeoutAndBeforeAfterHooks(t *testing.T) {
	defer goleak.VerifyNone(t)

	t.Run("timeout", func(t *testing.T) {
		conn := newFakeRealtimeConnection(4, 2)
		tool := Tool{Name: "slow", Execute: func(ctx context.Context, _ json.RawMessage) (string, error) {
			<-ctx.Done()
			return "", ctx.Err()
		}}
		session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn},
			WithRealtimeTools(tool), WithRealtimeToolTimeout(20*time.Millisecond))
		if err != nil {
			t.Fatal(err)
		}
		conn.events <- provider.RealtimeEvent{Type: provider.RealtimeToolCallEvent, ToolCall: &provider.RealtimeToolCall{ID: "timeout", Name: "slow"}}
		_ = receiveRealtimeEvent(t, session.Events())
		result := receiveRealtimeClientEvent(t, conn.sent).ToolResult
		if result == nil || !result.IsError || !json.Valid(result.Output) {
			t.Fatalf("timeout result = %+v", result)
		}
		if err := session.Close(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("hook transforms", func(t *testing.T) {
		conn := newFakeRealtimeConnection(4, 2)
		var executed atomic.Bool
		tool := Tool{Name: "tool", Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			executed.Store(true)
			if string(input) != `{"changed":true}` {
				t.Errorf("tool input = %s", input)
			}
			return "original", nil
		}}
		session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn},
			WithRealtimeTools(tool),
			WithRealtimeOnBeforeToolExecute(func(info BeforeToolExecuteInfo) BeforeToolExecuteResult {
				return BeforeToolExecuteResult{Input: json.RawMessage(`{"changed":true}`)}
			}),
			WithRealtimeOnAfterToolExecute(func(info AfterToolExecuteInfo) AfterToolExecuteResult {
				return AfterToolExecuteResult{Output: `{"changed":"output"}`}
			}),
		)
		if err != nil {
			t.Fatal(err)
		}
		conn.events <- provider.RealtimeEvent{Type: provider.RealtimeToolCallEvent, ToolCall: &provider.RealtimeToolCall{ID: "hook", Name: "tool", Input: json.RawMessage(`{}`)}}
		_ = receiveRealtimeEvent(t, session.Events())
		result := receiveRealtimeClientEvent(t, conn.sent).ToolResult
		if !executed.Load() || result == nil || string(result.Output) != `{"changed":"output"}` {
			t.Fatalf("hook result = %+v executed=%v", result, executed.Load())
		}
		if err := session.Close(); err != nil {
			t.Fatal(err)
		}
	})
}

type closeErrConnection struct {
	*fakeRealtimeConnection
	err error
}

func (c *closeErrConnection) Close() error {
	_ = c.fakeRealtimeConnection.Close()
	return c.err
}

func TestRealtimeSessionEdgeCases(t *testing.T) {
	defer goleak.VerifyNone(t)

	t.Run("send needs a context", func(t *testing.T) {
		conn := newFakeRealtimeConnection(1, 1)
		session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn})
		if err != nil {
			t.Fatal(err)
		}
		defer session.Close()
		var nilCtx context.Context
		if err := session.SendText(nilCtx, "x"); err == nil {
			t.Fatal("expected nil context error")
		}
	})

	t.Run("connection close error is reported", func(t *testing.T) {
		closeErr := errors.New("close failed")
		conn := &closeErrConnection{fakeRealtimeConnection: newFakeRealtimeConnection(1, 1), err: closeErr}
		session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn})
		if err != nil {
			t.Fatal(err)
		}
		go conn.finish(nil)
		if err := session.Close(); !errors.Is(err, closeErr) {
			t.Fatalf("Close = %v, want %v", err, closeErr)
		}
	})

	t.Run("cancelled context stops publishing", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		conn := newFakeRealtimeConnection(4, 1)
		// The hook runs right before an event is queued: cancelling there makes
		// the second event find both a full queue and a cancelled session.
		session, err := OpenRealtimeSession(ctx, &fakeRealtimeModel{id: "test", conn: conn},
			WithRealtimeEventBuffer(1), WithRealtimeBackpressureTimeout(time.Minute),
			WithRealtimeOnEvent(func(ev provider.RealtimeEvent) {
				if ev.Text == "two" {
					cancel()
				}
			}))
		if err != nil {
			t.Fatal(err)
		}
		conn.events <- provider.RealtimeEvent{Type: provider.RealtimeTextDelta, Text: "one"}
		conn.events <- provider.RealtimeEvent{Type: provider.RealtimeTextDelta, Text: "two"}
		waitRealtimeDone(t, session)
		if err := session.Err(); err != nil {
			t.Fatalf("cancellation must not be a session error: %v", err)
		}
		_ = session.Close()
	})
}

func TestEncodeRealtimeToolOutput(t *testing.T) {
	long := strings.Repeat("é", 600)
	got := string(encodeRealtimeToolOutput("", errors.New(long)))
	var decoded map[string]string
	if err := json.Unmarshal([]byte(got), &decoded); err != nil {
		t.Fatalf("invalid JSON %s", got)
	}
	if n := len([]rune(decoded["error"])); n != 503 || !strings.HasSuffix(decoded["error"], "...") {
		t.Fatalf("error message not truncated to 500 runes: %d", n)
	}
	if got := string(encodeRealtimeToolOutput(`{"ok":true}`, nil)); got != `{"ok":true}` {
		t.Fatalf("JSON output = %s", got)
	}
	if got := string(encodeRealtimeToolOutput("plain", nil)); got != `{"text":"plain"}` {
		t.Fatalf("text output = %s", got)
	}
}

func TestRealtimeCloneHelpers(t *testing.T) {
	event := provider.RealtimeClientEvent{
		Audio:           []byte{1},
		ProviderOptions: map[string]any{"list": []any{map[string]any{"k": "v"}}, "strings": []string{"a"}, "raw": json.RawMessage(`1`)},
		Cursor:          &provider.PlaybackCursor{ItemID: "i"},
		Context:         &provider.RealtimeContext{Text: "t"},
		ToolResult:      &provider.RealtimeToolResult{ToolCallID: "c", Output: json.RawMessage(`{}`)},
	}
	clone := cloneRealtimeClientEvent(event)
	event.Audio[0] = 9
	event.Context.Text = "changed"
	event.Cursor.ItemID = "changed"
	event.ToolResult.Output[0] = 'x'
	event.ProviderOptions["list"].([]any)[0].(map[string]any)["k"] = "changed"
	event.ProviderOptions["strings"].([]string)[0] = "changed"
	event.ProviderOptions["raw"].(json.RawMessage)[0] = '2'
	if clone.Audio[0] != 1 || clone.Cursor.ItemID != "i" || clone.Context.Text != "t" || string(clone.ToolResult.Output) != `{}` {
		t.Fatalf("client event aliased: %+v", clone)
	}
	if clone.ProviderOptions["list"].([]any)[0].(map[string]any)["k"] != "v" || clone.ProviderOptions["strings"].([]string)[0] != "a" || string(clone.ProviderOptions["raw"].(json.RawMessage)) != "1" {
		t.Fatalf("metadata aliased: %+v", clone.ProviderOptions)
	}
	delegation := provider.RealtimeEvent{Delegation: &provider.RealtimeDelegation{ID: "d"}}
	cloned := cloneRealtimeEvent(delegation)
	delegation.Delegation.ID = "changed"
	if cloned.Delegation.ID != "d" {
		t.Fatal("delegation aliased")
	}
	if cloneAnyMap(nil) != nil {
		t.Fatal("cloneAnyMap(nil) must stay nil")
	}
}

func TestRealtimeSessionParityMethodsSendCopies(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(1, 16)
	captured := make(chan provider.RealtimeParams, 1)
	model := &fakeRealtimeModel{id: "test", openFn: func(_ context.Context, params provider.RealtimeParams) (provider.RealtimeConnection, error) {
		captured <- params
		return conn, nil
	}}
	session, err := OpenRealtimeSession(t.Context(), model, WithRealtimeRawEvents(true))
	if err != nil {
		t.Fatal(err)
	}
	if params := <-captured; !params.IncludeRawEvents {
		t.Fatal("WithRealtimeRawEvents not applied")
	}

	frame := []byte{0xff, 0xd8}
	if err := session.InputImage(t.Context(), provider.RealtimeMedia{MIMEType: "image/jpeg", Data: frame}); err != nil {
		t.Fatal(err)
	}
	frame[0] = 0
	if got := receiveRealtimeClientEvent(t, conn.sent); got.Type != provider.RealtimeImageInput || got.Media == nil ||
		got.Media.MIMEType != "image/jpeg" || got.Media.Data[0] != 0xff {
		t.Fatalf("image event = %+v", got)
	}

	if err := session.EndInputAudio(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got := receiveRealtimeClientEvent(t, conn.sent); got.Type != provider.RealtimeInputAudioEnd {
		t.Fatalf("audio end event = %+v", got)
	}

	messages := []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}}
	if err := session.AppendConversation(t.Context(), provider.RealtimeConversation{Messages: messages, RequestResponse: true}); err != nil {
		t.Fatal(err)
	}
	messages[0].Content[0].Text = "changed"
	if got := receiveRealtimeClientEvent(t, conn.sent); got.Type != provider.RealtimeConversationAppend || got.Conversation == nil ||
		!got.Conversation.RequestResponse || got.Conversation.Messages[0].Content[0].Text != "hi" {
		t.Fatalf("conversation event = %+v", got)
	}

	output := json.RawMessage(`{"ok":true}`)
	result := provider.RealtimeToolResult{ToolCallID: "c1", Output: output, Partial: true,
		Media: []provider.RealtimeMedia{{MIMEType: "image/png", Data: []byte{1}}}}
	if err := session.SendToolResult(t.Context(), result); err != nil {
		t.Fatal(err)
	}
	output[0] = 'x'
	result.Media[0].Data[0] = 9
	got := receiveRealtimeClientEvent(t, conn.sent)
	if got.Type != provider.RealtimeToolResultEvent || got.ToolResult == nil || !got.ToolResult.Partial ||
		string(got.ToolResult.Output) != `{"ok":true}` || got.ToolResult.Media[0].Data[0] != 1 {
		t.Fatalf("tool result event = %+v", got)
	}

	raw := json.RawMessage(`{"custom":{}}`)
	if err := session.SendRaw(t.Context(), raw); err != nil {
		t.Fatal(err)
	}
	raw[0] = 'x'
	if got := receiveRealtimeClientEvent(t, conn.sent); got.Type != provider.RealtimeRawEvent || string(got.Raw) != `{"custom":{}}` {
		t.Fatalf("raw event = %+v", got)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRealtimeSessionDeclaredToolsReachTheApplication(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(4, 4)
	declared := Tool{Name: "approve", Description: "needs a human", InputSchema: json.RawMessage(`{"type":"object"}`)}
	run := Tool{Name: "run", Execute: func(context.Context, json.RawMessage) (string, error) { return `{"ran":true}`, nil }}
	session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn}, WithRealtimeTools(declared, run))
	if err != nil {
		t.Fatal(err)
	}

	conn.events <- provider.RealtimeEvent{Type: provider.RealtimeToolCallEvent, ToolCall: &provider.RealtimeToolCall{ID: "a", Name: "approve"}}
	if got := receiveRealtimeEvent(t, session.Events()); got.ToolCall == nil || got.ToolCall.Name != "approve" {
		t.Fatalf("declared tool call not delivered: %+v", got)
	}
	conn.events <- provider.RealtimeEvent{Type: provider.RealtimeToolCallEvent, ToolCall: &provider.RealtimeToolCall{ID: "r", Name: "run"}}
	_ = receiveRealtimeEvent(t, session.Events())

	// Only the executable tool is answered by the session.
	if got := receiveRealtimeClientEvent(t, conn.sent); got.ToolResult == nil || got.ToolResult.ToolCallID != "r" {
		t.Fatalf("unexpected automatic result: %+v", got)
	}
	select {
	case got := <-conn.sent:
		t.Fatalf("declared tool was answered automatically: %+v", got)
	case <-time.After(50 * time.Millisecond):
	}

	if err := session.SendToolResult(t.Context(), provider.RealtimeToolResult{ToolCallID: "a", Output: json.RawMessage(`{"approved":true}`)}); err != nil {
		t.Fatal(err)
	}
	if got := receiveRealtimeClientEvent(t, conn.sent); got.ToolResult == nil || got.ToolResult.ToolCallID != "a" {
		t.Fatalf("application result = %+v", got)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRealtimeSessionRejectsDuplicateToolNames(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(1, 1)
	model := &fakeRealtimeModel{id: "test", conn: conn}
	_, err := OpenRealtimeSession(t.Context(), model, WithRealtimeTools(Tool{Name: "same"}, Tool{Name: "same"}))
	if !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
		t.Fatalf("duplicate tools error = %v", err)
	}
}

func TestRealtimeSessionUpdateSession(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(8, 8)
	echo := func(name string) Tool {
		return Tool{Name: name, Execute: func(context.Context, json.RawMessage) (string, error) { return `{"tool":"` + name + `"}`, nil }}
	}
	session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn}, WithRealtimeTools(echo("old")))
	if err != nil {
		t.Fatal(err)
	}
	callTool := func(id, name string) *provider.RealtimeToolResult {
		t.Helper()
		conn.events <- provider.RealtimeEvent{Type: provider.RealtimeToolCallEvent, ToolCall: &provider.RealtimeToolCall{ID: id, Name: name}}
		_ = receiveRealtimeEvent(t, session.Events())
		return receiveRealtimeClientEvent(t, conn.sent).ToolResult
	}

	system := "be brief"
	options := map[string]any{"nested": map[string]any{"k": "v"}}
	if err := session.UpdateSession(t.Context(), RealtimeUpdate{System: &system, ProviderOptions: options}); err != nil {
		t.Fatal(err)
	}
	system = "changed"
	options["nested"].(map[string]any)["k"] = "changed"
	got := receiveRealtimeClientEvent(t, conn.sent)
	if got.Type != provider.RealtimeSessionUpdateEvent || got.Update == nil || *got.Update.System != "be brief" ||
		got.Update.Tools != nil || got.Update.ProviderOptions["nested"].(map[string]any)["k"] != "v" {
		t.Fatalf("update without tools = %+v", got.Update)
	}
	if result := callTool("1", "old"); result.IsError {
		t.Fatalf("tools changed by an update without tools: %s", result.Output)
	}

	if err := session.UpdateSession(t.Context(), RealtimeUpdate{Tools: []Tool{echo("new")}}); err != nil {
		t.Fatal(err)
	}
	got = receiveRealtimeClientEvent(t, conn.sent)
	if got.Update == nil || got.Update.System != nil || len(got.Update.Tools) != 1 || got.Update.Tools[0].Name != "new" {
		t.Fatalf("update with tools = %+v", got.Update)
	}
	if result := callTool("2", "new"); result.IsError || string(result.Output) != `{"tool":"new"}` {
		t.Fatalf("new tool result = %+v", result)
	}
	if result := callTool("3", "old"); !result.IsError {
		t.Fatalf("removed tool still runs: %s", result.Output)
	}

	if err := session.UpdateSession(t.Context(), RealtimeUpdate{Tools: []Tool{}}); err != nil {
		t.Fatal(err)
	}
	if got := receiveRealtimeClientEvent(t, conn.sent); got.Update == nil || got.Update.Tools == nil || len(got.Update.Tools) != 0 {
		t.Fatalf("removing all tools must send an empty, non-nil list: %+v", got.Update)
	}

	if err := session.UpdateSession(t.Context(), RealtimeUpdate{Tools: []Tool{{Name: "x"}, {Name: "x"}}}); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
		t.Fatalf("duplicate tools in update = %v", err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRealtimeSessionUpdateRestoresToolsWhenSendFails(t *testing.T) {
	defer goleak.VerifyNone(t)

	conn := newFakeRealtimeConnection(4, 4)
	run := Tool{Name: "run", Execute: func(context.Context, json.RawMessage) (string, error) { return `{}`, nil }}
	ran := make(chan error, 1)
	session, err := OpenRealtimeSession(t.Context(), &fakeRealtimeModel{id: "test", conn: conn},
		WithRealtimeTools(run),
		WithRealtimeOnToolCall(func(info ToolCallInfo) { ran <- info.Error }),
	)
	if err != nil {
		t.Fatal(err)
	}

	_ = conn.Close() // the provider rejects every send from now on
	if err := session.UpdateSession(t.Context(), RealtimeUpdate{Tools: []Tool{}}); !errors.Is(err, provider.ErrRealtimeSessionClosed) {
		t.Fatalf("update error = %v", err)
	}
	conn.events <- provider.RealtimeEvent{Type: provider.RealtimeToolCallEvent, ToolCall: &provider.RealtimeToolCall{ID: "r", Name: "run"}}
	_ = receiveRealtimeEvent(t, session.Events())
	select {
	case err := <-ran:
		if err != nil {
			t.Fatalf("the previous tools must stay active after a failed update: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("tool did not run")
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRealtimeParityCloneHelpers(t *testing.T) {
	system := "s"
	event := provider.RealtimeClientEvent{
		Raw:   json.RawMessage(`{}`),
		Media: &provider.RealtimeMedia{MIMEType: "image/jpeg", Data: []byte{1}},
		Conversation: &provider.RealtimeConversation{Messages: []provider.Message{{
			Role:            provider.RoleUser,
			ProviderOptions: map[string]any{"k": "v"},
			Content: []provider.Part{{
				Type: provider.PartToolCall, ToolInput: json.RawMessage(`{}`),
				ProviderOptions: map[string]any{"k": "v"}, RemoteRef: &provider.RemoteFileRef{ID: "f", Data: []byte{1}},
			}},
		}}},
		Update: &provider.RealtimeSessionUpdate{System: &system, Tools: []provider.ToolDefinition{}, ProviderOptions: map[string]any{"k": "v"}},
		ToolResult: &provider.RealtimeToolResult{
			Media:           []provider.RealtimeMedia{{Data: []byte{1}}},
			ProviderOptions: map[string]any{"k": "v"},
		},
	}
	clone := cloneRealtimeClientEvent(event)
	event.Raw[0] = 'x'
	event.Media.Data[0] = 9
	msg := &event.Conversation.Messages[0]
	msg.ProviderOptions["k"] = "changed"
	msg.Content[0].ToolInput[0] = 'x'
	msg.Content[0].ProviderOptions["k"] = "changed"
	msg.Content[0].RemoteRef.ID = "changed"
	msg.Content[0].RemoteRef.Data[0] = 9
	*event.Update.System = "changed"
	event.Update.ProviderOptions["k"] = "changed"
	event.ToolResult.Media[0].Data[0] = 9
	event.ToolResult.ProviderOptions["k"] = "changed"

	cmsg := clone.Conversation.Messages[0]
	if string(clone.Raw) != `{}` || clone.Media.Data[0] != 1 || cmsg.ProviderOptions["k"] != "v" ||
		string(cmsg.Content[0].ToolInput) != `{}` || cmsg.Content[0].ProviderOptions["k"] != "v" || cmsg.Content[0].RemoteRef.ID != "f" || cmsg.Content[0].RemoteRef.Data[0] != 1 {
		t.Fatalf("client event aliased: %+v", clone)
	}
	if *clone.Update.System != "s" || clone.Update.ProviderOptions["k"] != "v" || clone.Update.Tools == nil {
		t.Fatalf("update aliased or empty tools lost: %+v", clone.Update)
	}
	if clone.ToolResult.Media[0].Data[0] != 1 || clone.ToolResult.ProviderOptions["k"] != "v" {
		t.Fatalf("tool result aliased: %+v", clone.ToolResult)
	}
	if cloneToolDefinitions(nil) != nil || cloneRealtimeMessages(nil) != nil || cloneRealtimeMedia(nil) != nil {
		t.Fatal("nil inputs must stay nil")
	}

	serverEvent := provider.RealtimeEvent{
		Raw:     json.RawMessage(`{}`),
		Media:   &provider.RealtimeMedia{Data: []byte{1}},
		Sources: []provider.Source{{URL: "u", ProviderMetadata: map[string]any{"k": "v"}}},
	}
	cloned := cloneRealtimeEvent(serverEvent)
	serverEvent.Raw[0] = 'x'
	serverEvent.Media.Data[0] = 9
	serverEvent.Sources[0].URL = "changed"
	serverEvent.Sources[0].ProviderMetadata["k"] = "changed"
	if string(cloned.Raw) != `{}` || cloned.Media.Data[0] != 1 || cloned.Sources[0].URL != "u" || cloned.Sources[0].ProviderMetadata["k"] != "v" {
		t.Fatalf("event aliased: %+v", cloned)
	}
}
