package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/jsonmerge"
	"github.com/zendev-sh/goai/internal/openairealtime"
	"github.com/zendev-sh/goai/internal/realtimews"
	"github.com/zendev-sh/goai/provider"
)

var (
	_ provider.RealtimeModel        = (*liveModel)(nil)
	_ provider.RealtimeCapableModel = (*liveModel)(nil)
	_ provider.RealtimeConnection   = (*liveConn)(nil)
)

const (
	liveReadLimit    = 16 << 20
	liveEventBuffer  = 64
	liveMaxErrorBody = 1024
)

// liveCloseTimeout bounds the wait for session.closed, which carries the final
// usage; OpenAI's reference client waits 15 seconds.
var liveCloseTimeout = 15 * time.Second

var (
	livePCM24k = provider.AudioFormat{Encoding: provider.AudioPCM16, SampleRate: 24_000, Channels: 1}
	livePCM16k = provider.AudioFormat{Encoding: provider.AudioPCM16, SampleRate: 16_000, Channels: 1}
	livePCMU   = provider.AudioFormat{Encoding: provider.AudioG711U, SampleRate: 8_000, Channels: 1}
	livePCMA   = provider.AudioFormat{Encoding: provider.AudioG711A, SampleRate: 8_000, Channels: 1}
)

// LiveResponsesDelegation returns ProviderOptions that make GPT-Live delegate
// reasoning and tools to a Responses model, such as "gpt-6-luna". Session tools
// run on that backend; GoAI executes the function tools it calls. Add other
// delegation.responses fields (instructions, reasoning, hosted tools) to the
// returned map before passing it to WithRealtimeProviderOptions.
func LiveResponsesDelegation(model string) map[string]any {
	return map[string]any{"delegation": map[string]any{
		"type":      "responses",
		"responses": map[string]any{"model": model},
	}}
}

// LiveClientDelegation returns ProviderOptions that hand delegated work to the
// application: it receives RealtimeDelegationCreated events and answers with
// RealtimeSession.AppendContext. This is the default.
func LiveClientDelegation() map[string]any {
	return map[string]any{"delegation": map[string]any{"type": "client"}}
}

// Live creates an OpenAI GPT-Live model, such as "gpt-live-1": a full-duplex
// voice model that hands reasoning and tool use to a backend. It speaks the
// Live API over WebSocket from the backend and reuses the provider options
// for the API key, base URL, headers, and HTTP client.
//
// Delegation is chosen at session start with LiveClientDelegation (default) or
// LiveResponsesDelegation. Other session fields go in ProviderOptions using
// their Live API names. Close sends session.close and waits for
// session.closed so the final usage is reported.
//
// Experimental: see provider.RealtimeModel.
func Live(modelID string, opts ...Option) provider.RealtimeModel {
	return &liveModel{id: modelID, opts: resolveConnectOptions(opts)}
}

// LiveFork starts a new session from a stored GPT-Live session over
// WebSocket. It inherits the model, instructions, voice, and delegation;
// ProviderOptions may set "store" and override the Responses backend, and the
// audio format applies to the new connection only. ModelID reports the source
// session ID.
//
// Experimental: see provider.RealtimeModel.
func LiveFork(sourceSessionID string, opts ...Option) provider.RealtimeModel {
	return &liveModel{id: sourceSessionID, opts: resolveConnectOptions(opts), mode: liveForkMode}
}

// LiveSideband attaches the backend to an existing GPT-Live session, such as
// one whose audio runs over WebRTC or SIP, to observe and steer it: it
// receives the session events and sends updates, context, and tool results,
// but no audio. Attaching starts nothing; declare the session's audio format
// with WithRealtimeAudio to label output audio. ModelID reports the session ID.
//
// Experimental: see provider.RealtimeModel.
func LiveSideband(sessionID string, opts ...Option) provider.RealtimeModel {
	return &liveModel{id: sessionID, opts: resolveConnectOptions(opts), mode: liveSidebandMode}
}

type liveMode int

const (
	livePrimaryMode liveMode = iota
	liveForkMode
	liveSidebandMode
)

type liveModel struct {
	id   string
	opts options
	mode liveMode
}

func (m *liveModel) ModelID() string { return m.id }

// RealtimeCapabilities describes GPT-Live over WebSocket. The model manages
// turns and interruptions itself; there is no manual turn control.
func (m *liveModel) RealtimeCapabilities() provider.RealtimeCapabilities {
	formats := []provider.AudioFormat{livePCM24k, livePCM16k, livePCMU, livePCMA}
	return provider.RealtimeCapabilities{
		InputAudio:           m.mode != liveSidebandMode,
		OutputAudio:          true,
		InputTranscription:   true,
		OutputTranscription:  true,
		TextInput:            true,
		ImageInput:           true,
		ProviderVAD:          true,
		BargeIn:              true,
		FunctionCalling:      true,
		NonBlockingToolCalls: true,
		ParallelToolCalls:    true,
		RuntimeSessionUpdate: true,
		ConversationAppend:   true,
		UsageUpdates:         true,
		Delegation:           true,
		RawEvents:            true,
		SupportedInputAudio:  formats,
		SupportedOutputAudio: append([]provider.AudioFormat(nil), formats...),
	}
}

// OpenRealtime dials the Live endpoint of the model's mode. A new session or
// fork sends session.start and returns once the server reports
// session.started; a sideband is ready as soon as it attaches.
func (m *liveModel) OpenRealtime(ctx context.Context, params provider.RealtimeParams) (provider.RealtimeConnection, error) {
	if err := params.Validate(); err != nil {
		return nil, err
	}
	var (
		session    map[string]any
		format     provider.AudioFormat
		delegation string
		path       = "/live/sessions"
		err        error
	)
	switch m.mode {
	case liveForkMode:
		session, format, err = buildLiveForkSession(params)
		path += "/" + url.PathEscape(m.id) + "/fork"
	case liveSidebandMode:
		format, err = liveSidebandFormat(params)
		path += "/" + url.PathEscape(m.id) + "/attach"
	default:
		session, format, delegation, err = buildLiveSession(m.id, params)
	}
	if err != nil {
		return nil, err
	}
	if m.opts.tokenSource == nil {
		return nil, errors.New("goai: no API key or token source configured")
	}
	token, err := m.opts.tokenSource.Token(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}
	endpoint, err := liveURL(m.opts.baseURL, path)
	if err != nil {
		return nil, err
	}

	header := http.Header{}
	for k, v := range m.opts.headers {
		header.Set(k, v)
	}
	header.Set("Authorization", "Bearer "+token)

	ws, resp, err := realtimews.Dial(ctx, endpoint, realtimews.Options{
		HTTPClient: m.opts.httpClient,
		Headers:    header,
		ReadLimit:  liveReadLimit,
	})
	if err != nil {
		return nil, liveDialError(err, resp)
	}

	connCtx, cancel := context.WithCancel(ctx)
	c := &liveConn{
		ws:         ws,
		ctx:        connCtx,
		cancel:     cancel,
		events:     make(chan provider.RealtimeEvent, liveEventBuffer),
		done:       make(chan struct{}),
		closing:    make(chan struct{}),
		closed:     make(chan struct{}),
		format:     format,
		delegation: delegation,
		sideband:   m.mode == liveSidebandMode,
		inputTr:    params.InputTranscription.Enabled,
		outputTr:   params.OutputTranscription.Enabled,
		pending:    make(map[string]bool),
		rawEvents:  params.IncludeRawEvents,
	}
	sessionID := m.id
	if m.mode != liveSidebandMode {
		info, err := c.start(ctx, session)
		if err != nil {
			c.abort()
			return nil, err
		}
		sessionID = c.adopt(info)
	}
	c.setup = append(c.setup, provider.RealtimeEvent{Type: provider.RealtimeSessionReady, SessionID: sessionID})
	go c.readLoop()
	return c, nil
}

// adopt takes the delegation and audio format the server reports, which a
// fork inherits from its source, and returns the session ID.
func (c *liveConn) adopt(info *liveSessionInfo) string {
	if info == nil {
		return ""
	}
	if info.Delegation != nil && info.Delegation.Type != "" {
		c.delegation = info.Delegation.Type
	}
	if info.Audio != nil && info.Audio.Format != nil {
		if f, ok := liveFormatFromWire(info.Audio.Format.Type, info.Audio.Format.Rate); ok {
			c.format = f
		}
	}
	return info.ID
}

func liveFormatFromWire(typ string, rate int) (provider.AudioFormat, bool) {
	switch {
	case typ == "audio/pcm" && rate == 16_000:
		return livePCM16k, true
	case typ == "audio/pcm":
		return livePCM24k, true
	case typ == "audio/pcmu":
		return livePCMU, true
	case typ == "audio/pcma":
		return livePCMA, true
	}
	return provider.AudioFormat{}, false
}

func liveURL(base, path string) (string, error) {
	u, err := url.Parse(strings.TrimRight(base, "/"))
	if err != nil {
		return "", fmt.Errorf("openai: invalid base URL: %w", err)
	}
	switch u.Scheme {
	case "https", "wss":
		u.Scheme = "wss"
	case "http", "ws":
		u.Scheme = "ws"
	default:
		return "", fmt.Errorf("openai: unsupported base URL scheme %q for Live", u.Scheme)
	}
	u.Path += path
	return u.String(), nil
}

func liveDialError(err error, resp *http.Response) error {
	if resp == nil || resp.StatusCode == http.StatusSwitchingProtocols {
		return fmt.Errorf("openai: connecting to live api: %w", err)
	}
	var body []byte
	if resp.Body != nil {
		body, _ = io.ReadAll(io.LimitReader(resp.Body, liveMaxErrorBody))
		_ = resp.Body.Close()
	}
	return goai.ParseHTTPErrorWithHeaders("openai", resp.StatusCode, body, resp.Header)
}

// validateLiveParams rejects settings GPT-Live has no place for: the model
// manages turns itself and its transcripts take no configuration.
func validateLiveParams(params provider.RealtimeParams) error {
	if params.Resumption != nil {
		return fmt.Errorf("%w: GPT-Live has no session resumption; fork a stored session instead", provider.ErrRealtimeUnsupported)
	}
	td := params.TurnDetection
	if (td.Mode != "" && td.Mode != provider.TurnModeProviderVAD) || td.DisableAutoResponse || td.DisableInterruption || len(td.ProviderOptions) > 0 {
		return fmt.Errorf("%w: GPT-Live manages turns and interruptions itself", provider.ErrRealtimeUnsupported)
	}
	for _, tc := range []provider.TranscriptionConfig{params.InputTranscription, params.OutputTranscription} {
		if tc.Language != "" || tc.Prompt != "" || len(tc.ProviderOptions) > 0 {
			return fmt.Errorf("%w: GPT-Live transcripts take no configuration", provider.ErrRealtimeUnsupported)
		}
	}
	for key := range params.ProviderOptions {
		if key == "model" || key == "instructions" {
			return fmt.Errorf("%w: %s is derived from the realtime params", provider.ErrInvalidRealtimeConfig, key)
		}
	}
	return nil
}

// buildLiveSession maps RealtimeParams to the session object of session.start.
// It returns the shared audio format and the delegation type.
func buildLiveSession(modelID string, params provider.RealtimeParams) (map[string]any, provider.AudioFormat, string, error) {
	format, wire, err := liveFormat(params.InputAudio, params.OutputAudio)
	if err != nil {
		return nil, provider.AudioFormat{}, "", err
	}
	if err := validateLiveParams(params); err != nil {
		return nil, provider.AudioFormat{}, "", err
	}

	audio := map[string]any{"format": wire}
	if params.Voice != "" {
		audio["output"] = map[string]any{"voice": params.Voice}
	}
	session := map[string]any{
		"model":      modelID,
		"audio":      audio,
		"delegation": map[string]any{"type": "client"},
	}
	if params.System != "" {
		session["instructions"] = params.System
	}
	session = jsonmerge.Deep(session, params.ProviderOptions)
	if delegation, _ := session["delegation"].(map[string]any); delegation["type"] == "responses" && delegation["responses"] == nil {
		return nil, provider.AudioFormat{}, "", fmt.Errorf("%w: responses delegation needs a backend model", provider.ErrInvalidRealtimeConfig)
	}
	session, kind, err := withLiveBackend(session, params)
	return session, format, kind, err
}

// buildLiveForkSession builds the session.start of a fork: everything not set
// is inherited from the stored session, which cannot change model,
// instructions, or voice.
func buildLiveForkSession(params provider.RealtimeParams) (map[string]any, provider.AudioFormat, error) {
	if params.System != "" || params.Voice != "" {
		return nil, provider.AudioFormat{}, fmt.Errorf("%w: a fork inherits the instructions and voice of its source", provider.ErrRealtimeUnsupported)
	}
	if err := validateLiveParams(params); err != nil {
		return nil, provider.AudioFormat{}, err
	}
	session := map[string]any{}
	var format provider.AudioFormat
	if !params.InputAudio.IsZero() || !params.OutputAudio.IsZero() {
		f, wire, err := liveFormat(params.InputAudio, params.OutputAudio)
		if err != nil {
			return nil, provider.AudioFormat{}, err
		}
		format = f
		session["audio"] = map[string]any{"format": wire}
	}
	session = jsonmerge.Deep(session, params.ProviderOptions)
	if len(params.Tools) > 0 || params.ToolChoice != "" || params.MaxOutputTokens > 0 {
		// Backend overrides apply only to a source with Responses delegation.
		session = jsonmerge.Deep(session, map[string]any{"delegation": map[string]any{"type": "responses"}})
	}
	session, _, err := withLiveBackend(session, params)
	return session, format, err
}

// liveSidebandFormat is the format the application declares for the attached
// session; the sideband carries no audio of its own.
func liveSidebandFormat(params provider.RealtimeParams) (provider.AudioFormat, error) {
	if params.System != "" || params.Voice != "" || len(params.Tools) > 0 || params.ToolChoice != "" ||
		params.MaxOutputTokens > 0 || len(params.ProviderOptions) > 0 {
		return provider.AudioFormat{}, fmt.Errorf("%w: a sideband attaches to a running session; change it with UpdateSession", provider.ErrRealtimeUnsupported)
	}
	if err := validateLiveParams(params); err != nil {
		return provider.AudioFormat{}, err
	}
	if params.InputAudio.IsZero() && params.OutputAudio.IsZero() {
		return provider.AudioFormat{}, nil
	}
	format, _, err := liveFormat(params.InputAudio, params.OutputAudio)
	return format, err
}

// withLiveBackend adds tools, tool choice, and the token limit to the
// Responses backend. Without Responses delegation there is no place for them.
func withLiveBackend(session map[string]any, params provider.RealtimeParams) (map[string]any, string, error) {
	delegation, _ := session["delegation"].(map[string]any)
	kind, _ := delegation["type"].(string)
	if kind != "responses" {
		if len(params.Tools) > 0 || params.ToolChoice != "" || params.MaxOutputTokens > 0 {
			return nil, "", fmt.Errorf("%w: GPT-Live runs tools on a Responses backend; use LiveResponsesDelegation", provider.ErrRealtimeUnsupported)
		}
		return session, kind, nil
	}
	responses, _ := delegation["responses"].(map[string]any)
	backend := make(map[string]any, len(responses)+3)
	for k, v := range responses {
		backend[k] = v
	}
	if len(params.Tools) > 0 {
		existing, _ := backend["tools"].([]any)
		tools, err := liveTools(params.Tools)
		if err != nil {
			return nil, "", err
		}
		backend["tools"] = append(append([]any(nil), existing...), tools...)
	}
	if params.ToolChoice != "" {
		backend["tool_choice"] = toolChoiceValue(params.ToolChoice)
	}
	if params.MaxOutputTokens > 0 {
		backend["max_output_tokens"] = params.MaxOutputTokens
	}
	session["delegation"] = jsonmerge.Deep(delegation, map[string]any{"responses": backend})
	return session, kind, nil
}

// liveTools maps tools for the Responses backend: functions, and hosted tools
// (web search, MCP, file search, and the rest) in the Responses API format.
func liveTools(defs []provider.ToolDefinition) ([]any, error) {
	tools := make([]any, 0, len(defs))
	for _, def := range defs {
		if def.ProviderDefinedType != "" {
			tool := map[string]any{"type": def.ProviderDefinedType}
			for k, v := range def.ProviderDefinedOptions {
				tool[k] = v
			}
			tools = append(tools, tool)
			continue
		}
		schema := json.RawMessage(`{"type":"object","properties":{}}`)
		if len(def.InputSchema) > 0 {
			if !json.Valid(def.InputSchema) {
				return nil, fmt.Errorf("%w: invalid schema for tool %s", provider.ErrInvalidRealtimeConfig, def.Name)
			}
			schema = def.InputSchema
		}
		tools = append(tools, map[string]any{"type": "function", "name": def.Name, "description": def.Description, "parameters": schema})
	}
	return tools, nil
}

func toolChoiceValue(choice string) any {
	switch choice {
	case "auto", "none", "required":
		return choice
	default:
		return map[string]any{"type": "function", "name": choice}
	}
}

// liveFormat resolves the single audio format GPT-Live uses for both directions.
func liveFormat(in, out provider.AudioFormat) (provider.AudioFormat, map[string]any, error) {
	f := in
	if f.IsZero() {
		f = out
	}
	if !in.IsZero() && !out.IsZero() && in != out {
		return provider.AudioFormat{}, nil, fmt.Errorf("%w: GPT-Live uses one audio format for input and output", provider.ErrUnsupportedAudioFormat)
	}
	switch f {
	case provider.AudioFormat{}, livePCM24k:
		return livePCM24k, map[string]any{"type": "audio/pcm", "rate": 24_000}, nil
	case livePCM16k:
		return livePCM16k, map[string]any{"type": "audio/pcm", "rate": 16_000}, nil
	case livePCMU:
		return livePCMU, map[string]any{"type": "audio/pcmu", "rate": 8_000}, nil
	case livePCMA:
		return livePCMA, map[string]any{"type": "audio/pcma", "rate": 8_000}, nil
	}
	return provider.AudioFormat{}, nil, fmt.Errorf("%w: GPT-Live takes 24 or 16 kHz mono PCM16 or 8 kHz G.711", provider.ErrUnsupportedAudioFormat)
}

// liveConn is one GPT-Live WebSocket session. One reader goroutine translates
// server events; writes may come from any goroutine.
type liveConn struct {
	ws      *realtimews.Conn
	ctx     context.Context
	cancel  context.CancelFunc
	events  chan provider.RealtimeEvent
	done    chan struct{}
	closing chan struct{}
	closed  chan struct{} // session.closed received

	closeOnce  sync.Once
	closedOnce sync.Once

	errMu sync.Mutex
	err   error

	format     provider.AudioFormat
	delegation string // empty when unknown, as on a sideband
	sideband   bool
	inputTr    bool
	outputTr   bool
	rawEvents  bool

	// Responses delegation: once every pending function call has its output,
	// a single response.create continues the backend.
	pendingMu sync.Mutex
	pending   map[string]bool

	// Reader goroutine only. setup holds events produced while starting,
	// delivered before anything the reader reads.
	setup []provider.RealtimeEvent
	seq   int64
}

func (c *liveConn) Events() <-chan provider.RealtimeEvent { return c.events }

func (c *liveConn) Err() error {
	c.errMu.Lock()
	defer c.errMu.Unlock()
	return c.err
}

func (c *liveConn) setErr(err error) {
	if err == nil {
		return
	}
	c.errMu.Lock()
	if c.err == nil {
		c.err = err
	}
	c.errMu.Unlock()
}

// Close sends session.close, waits up to liveCloseTimeout for session.closed
// (its final usage is still delivered when the queue has room), then closes
// the WebSocket. Idempotent.
func (c *liveConn) Close() error {
	c.closeOnce.Do(func() {
		close(c.closing)
		ctx, cancel := context.WithTimeout(context.Background(), liveCloseTimeout)
		defer cancel()
		if c.write(ctx, map[string]any{"type": "session.close"}) == nil {
			select {
			case <-c.closed:
			case <-c.done:
			case <-ctx.Done():
			}
		}
		_ = c.ws.Close()
		c.cancel()
	})
	<-c.done
	return nil
}

func (c *liveConn) abort() {
	c.closeOnce.Do(func() {
		_ = c.ws.CloseNow()
		c.cancel()
		close(c.events)
		close(c.done)
	})
}

func (c *liveConn) start(ctx context.Context, session map[string]any) (*liveSessionInfo, error) {
	if err := c.write(ctx, map[string]any{"type": "session.start", "session": session}); err != nil {
		return nil, fmt.Errorf("openai: sending session.start: %w", err)
	}
	for {
		ev, data, err := c.read(ctx)
		if err != nil {
			return nil, err
		}
		switch ev.Type {
		case "session.started":
			if c.rawEvents {
				c.setup = append(c.setup, liveRawEvent(data))
			}
			return ev.Session, nil
		case "error":
			return nil, liveError(ev.Error)
		}
		c.setup = append(c.setup, c.translateMessage(ev, data)...)
	}
}

func (c *liveConn) read(ctx context.Context) (liveServerEvent, []byte, error) {
	_, data, err := c.ws.Read(ctx)
	if err != nil {
		code, reason, ok := realtimews.CloseStatus(err)
		switch {
		case !ok:
			return liveServerEvent{}, nil, fmt.Errorf("openai: live connection lost: %w", err)
		case code == realtimews.StatusNormalClosure:
			return liveServerEvent{}, nil, errLiveClosedNormally
		default:
			return liveServerEvent{}, nil, fmt.Errorf("openai: live session closed with status %d: %s", code, reason)
		}
	}
	var ev liveServerEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return liveServerEvent{}, nil, fmt.Errorf("openai: decoding live event: %w", err)
	}
	return ev, data, nil
}

// translateMessage adds the raw event to its normalized events: always with
// IncludeRawEvents, otherwise when the adapter does not know its type.
func (c *liveConn) translateMessage(ev liveServerEvent, data []byte) []provider.RealtimeEvent {
	events, known := c.translate(ev)
	if !c.rawEvents && known {
		return events
	}
	return append([]provider.RealtimeEvent{liveRawEvent(data)}, events...)
}

func liveRawEvent(data []byte) provider.RealtimeEvent {
	return provider.RealtimeEvent{Type: provider.RealtimeProviderEvent, Raw: append(json.RawMessage(nil), data...)}
}

var errLiveClosedNormally = errors.New("openai: live connection closed")

func liveError(e *liveWireError) error {
	if e == nil {
		return errors.New("openai: live error")
	}
	return fmt.Errorf("openai: live %s (%s): %s", cmpString(e.Code, e.Type), e.Type, e.Message)
}

func cmpString(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func (c *liveConn) write(ctx context.Context, ev map[string]any) error {
	data, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	return c.ws.Write(ctx, realtimews.MessageText, data)
}

func (c *liveConn) Send(ctx context.Context, event provider.RealtimeClientEvent) error {
	if c.ctx.Err() != nil {
		return provider.ErrRealtimeSessionClosed
	}
	select {
	case <-c.closing:
		return provider.ErrRealtimeSessionClosed
	default:
	}
	if event.Type == provider.RealtimeRawEvent {
		if !realtimews.IsJSONObject(event.Raw) {
			return fmt.Errorf("%w: raw event must be a JSON object", provider.ErrInvalidRealtimeConfig)
		}
		return c.sendError(c.ws.Write(ctx, realtimews.MessageText, event.Raw))
	}
	msgs, err := c.encode(event)
	if err != nil {
		return err
	}
	for _, msg := range msgs {
		if err := c.writeEvent(ctx, msg); err != nil {
			return err
		}
	}
	// The backend continues once every pending call has its output.
	if event.Type == provider.RealtimeToolResultEvent && c.outputSent(event.ToolResult.ToolCallID) {
		return c.writeEvent(ctx, map[string]any{"type": "response.create"})
	}
	return nil
}

func (c *liveConn) writeEvent(ctx context.Context, msg map[string]any) error {
	return c.sendError(c.write(ctx, msg))
}

// sendError maps a failed write to the error a sender sees.
func (c *liveConn) sendError(err error) error {
	if err == nil {
		return nil
	}
	if c.ctx.Err() != nil {
		return provider.ErrRealtimeSessionClosed
	}
	return fmt.Errorf("openai: sending live event: %w", err)
}

func (c *liveConn) outputSent(callID string) bool {
	c.pendingMu.Lock()
	defer c.pendingMu.Unlock()
	if !c.pending[callID] {
		return false
	}
	delete(c.pending, callID)
	return len(c.pending) == 0
}

// encode maps a client event by delegation: the Responses backend takes items
// and response.create, the Decisions backend takes decision items and
// decision.create, and with client delegation that work runs in the app. An
// unknown delegation (a sideband) leaves the check to the server.
func (c *liveConn) encode(event provider.RealtimeClientEvent) ([]map[string]any, error) {
	switch event.Type {
	case provider.RealtimeInputAudioAppend:
		if c.sideband {
			return nil, fmt.Errorf("%w: the session's audio runs on its own transport, not the sideband", provider.ErrRealtimeUnsupported)
		}
		if len(event.Audio) == 0 {
			return nil, fmt.Errorf("%w: empty audio frame", provider.ErrInvalidRealtimeConfig)
		}
		return []map[string]any{{"type": "session.input_audio.append", "audio": base64.StdEncoding.EncodeToString(event.Audio)}}, nil
	case provider.RealtimeContextAppend:
		return c.encodeContext(event.Context)
	case provider.RealtimeToolResultEvent:
		return c.encodeToolOutput(event.ToolResult)
	case provider.RealtimeTextInput:
		return c.encodeUserInput(map[string]any{"type": "input_text", "text": event.Text}, true)
	case provider.RealtimeImageInput:
		media := event.Media
		if media == nil || !strings.HasPrefix(media.MIMEType, "image/") || len(media.Data) == 0 {
			return nil, fmt.Errorf("%w: image input needs an image MIME type and data", provider.ErrInvalidRealtimeConfig)
		}
		url := "data:" + media.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(media.Data)
		return c.encodeUserInput(map[string]any{"type": "input_image", "image_url": url}, false)
	case provider.RealtimeConversationAppend:
		return c.encodeConversation(event.Conversation)
	case provider.RealtimeSessionUpdateEvent:
		return c.encodeSessionUpdate(event.Update)
	case provider.RealtimeResponseCreate:
		if len(event.ProviderOptions) > 0 {
			return nil, fmt.Errorf("%w: GPT-Live response.create takes no parameters", provider.ErrRealtimeUnsupported)
		}
		create, err := c.backendCreate()
		if err != nil {
			return nil, err
		}
		return []map[string]any{create}, nil
	default:
		return nil, fmt.Errorf("%w: GPT-Live does not support %s", provider.ErrRealtimeUnsupported, event.Type)
	}
}

// backendCreate asks the delegated backend to run now.
func (c *liveConn) backendCreate() (map[string]any, error) {
	switch c.delegation {
	case "decisions":
		return map[string]any{"type": "decision.create"}, nil
	case "responses", "":
		return map[string]any{"type": "response.create"}, nil
	default:
		return nil, fmt.Errorf("%w: with client delegation the backend is your own", provider.ErrRealtimeUnsupported)
	}
}

// encodeUserInput adds one user content part to the backend; typed text also
// asks the backend to respond, as typing a message would.
func (c *liveConn) encodeUserInput(part map[string]any, respond bool) ([]map[string]any, error) {
	item, err := c.backendItem(map[string]any{"type": "message", "role": "user", "content": []map[string]any{part}})
	if err != nil {
		return nil, err
	}
	msgs := []map[string]any{item}
	if respond {
		create, _ := c.backendCreate()
		msgs = append(msgs, create)
	}
	return msgs, nil
}

// backendItem wraps an item for the delegated backend.
func (c *liveConn) backendItem(item map[string]any) (map[string]any, error) {
	switch c.delegation {
	case "decisions":
		if item["type"] != "message" || item["role"] != "user" {
			return nil, fmt.Errorf("%w: the Decisions backend takes user messages only", provider.ErrInvalidRealtimeConfig)
		}
		return map[string]any{"type": "decision.item.create", "item": item}, nil
	case "responses", "":
		return map[string]any{"type": "response.item.create", "item": item}, nil
	default:
		return nil, fmt.Errorf("%w: with client delegation, input goes to your own backend", provider.ErrRealtimeUnsupported)
	}
}

func (c *liveConn) encodeConversation(conv *provider.RealtimeConversation) ([]map[string]any, error) {
	if conv == nil || len(conv.Messages) == 0 {
		return nil, fmt.Errorf("%w: conversation needs messages", provider.ErrInvalidRealtimeConfig)
	}
	items, err := openairealtime.ConversationItems(conv.Messages)
	if err != nil {
		return nil, err
	}
	msgs := make([]map[string]any, 0, len(items)+1)
	for _, item := range items {
		msg, err := c.backendItem(item)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, msg)
	}
	if conv.RequestResponse {
		create, _ := c.backendCreate()
		msgs = append(msgs, create)
	}
	return msgs, nil
}

// encodeToolOutput answers a backend function call. The output is one text
// value, so partial results and media have no place in it.
func (c *liveConn) encodeToolOutput(result *provider.RealtimeToolResult) ([]map[string]any, error) {
	if result == nil || result.ToolCallID == "" {
		return nil, fmt.Errorf("%w: tool result needs a tool call ID", provider.ErrInvalidRealtimeConfig)
	}
	if c.delegation != "responses" && c.delegation != "" {
		return nil, fmt.Errorf("%w: tool results need Responses delegation", provider.ErrRealtimeUnsupported)
	}
	if result.Partial || len(result.Media) > 0 {
		return nil, fmt.Errorf("%w: GPT-Live tool outputs are a single text result", provider.ErrRealtimeUnsupported)
	}
	for key := range result.ProviderOptions {
		if key == "type" || key == "call_id" {
			return nil, fmt.Errorf("%w: %s is derived from the tool result", provider.ErrInvalidRealtimeConfig, key)
		}
	}
	item := jsonmerge.Deep(map[string]any{
		"type":    "function_call_output",
		"call_id": result.ToolCallID,
		"output":  string(result.Output),
	}, result.ProviderOptions)
	return []map[string]any{{"type": "response.item.create", "item": item}}, nil
}

// encodeSessionUpdate maps an update to session.update, which changes the
// delegation backend and triggers; instructions are fixed at start and grow
// through AppendContext.
func (c *liveConn) encodeSessionUpdate(update *provider.RealtimeSessionUpdate) ([]map[string]any, error) {
	if update == nil {
		return nil, fmt.Errorf("%w: empty session update", provider.ErrInvalidRealtimeConfig)
	}
	if update.System != nil {
		return nil, fmt.Errorf("%w: GPT-Live instructions are fixed; add more with AppendContext", provider.ErrRealtimeUnsupported)
	}
	session := map[string]any{}
	if update.Tools != nil {
		if c.delegation != "responses" && c.delegation != "" {
			return nil, fmt.Errorf("%w: GPT-Live runs tools on a Responses backend", provider.ErrRealtimeUnsupported)
		}
		tools, err := liveTools(update.Tools)
		if err != nil {
			return nil, err
		}
		session["delegation"] = map[string]any{"type": "responses", "responses": map[string]any{"tools": tools}}
	}
	for key := range update.ProviderOptions {
		if key == "model" || key == "instructions" {
			return nil, fmt.Errorf("%w: %s cannot change after session start", provider.ErrInvalidRealtimeConfig, key)
		}
	}
	session = jsonmerge.Deep(session, update.ProviderOptions)
	if len(session) == 0 {
		return nil, fmt.Errorf("%w: empty session update", provider.ErrInvalidRealtimeConfig)
	}
	return []map[string]any{{"type": "session.update", "session": session}}, nil
}

func (c *liveConn) encodeContext(content *provider.RealtimeContext) ([]map[string]any, error) {
	if content == nil || content.Text == "" {
		return nil, fmt.Errorf("%w: context needs text", provider.ErrInvalidRealtimeConfig)
	}
	var typ string
	switch content.Kind {
	case provider.RealtimeContextCommentary:
		typ = "session.commentary.append"
	case provider.RealtimeContextThinking:
		typ = "session.thinking.append"
	case provider.RealtimeContextInstructions:
		typ = "session.instructions.append"
	default:
		return nil, fmt.Errorf("%w: unknown context kind %q", provider.ErrInvalidRealtimeConfig, content.Kind)
	}
	var delegationID any // null: session-wide context
	if content.DelegationID != "" {
		delegationID = content.DelegationID
	}
	return []map[string]any{{"type": typ, "delegation_id": delegationID, "content": content.Text}}, nil
}

func (c *liveConn) readLoop() {
	defer close(c.done)
	defer close(c.events)
	for _, event := range c.setup {
		c.seq++
		event.Sequence = c.seq
		if !c.emit(event) {
			return
		}
	}
	c.setup = nil
	for {
		ev, data, err := c.read(c.ctx)
		if err != nil {
			if c.ctx.Err() == nil && !errors.Is(err, errLiveClosedNormally) {
				select {
				case <-c.closing: // ending after our session.close is expected
				default:
					c.setErr(err)
				}
			}
			return
		}
		for _, event := range c.translateMessage(ev, data) {
			c.seq++
			event.Sequence = c.seq
			if !c.emit(event) {
				return
			}
		}
		if ev.Type == "session.closed" {
			c.closedOnce.Do(func() { close(c.closed) })
		}
	}
}

// emit queues an event. After Close it only uses free queue space, so a
// consumer that stopped reading cannot block the close handshake.
func (c *liveConn) emit(event provider.RealtimeEvent) bool {
	select {
	case c.events <- event:
		return true
	default:
	}
	select {
	case c.events <- event:
		return true
	case <-c.closing:
		return true
	case <-c.ctx.Done():
		return false
	}
}

// translate maps one server event to common events and reports whether the
// adapter knows it. Reader goroutine only.
func (c *liveConn) translate(ev liveServerEvent) ([]provider.RealtimeEvent, bool) {
	switch ev.Type {
	case "session.output_audio.delta":
		audio, err := base64.StdEncoding.DecodeString(ev.Delta)
		if err != nil {
			return []provider.RealtimeEvent{{Type: provider.RealtimeError, Error: fmt.Errorf("openai: decoding live audio: %w", err)}}, true
		}
		return []provider.RealtimeEvent{{Type: provider.RealtimeOutputAudio, Audio: audio, AudioFormat: c.format}}, true
	case "session.input_transcript.delta":
		if !c.inputTr {
			return nil, true
		}
		return []provider.RealtimeEvent{{Type: provider.RealtimeInputTranscript, Text: ev.Delta, AudioStart: liveMs(ev.StartMs), AudioEnd: liveMs(ev.EndMs)}}, true
	case "session.output_transcript.delta":
		if !c.outputTr {
			return nil, true
		}
		return []provider.RealtimeEvent{{Type: provider.RealtimeOutputTranscript, Text: ev.Delta, AudioStart: liveMs(ev.StartMs), AudioEnd: liveMs(ev.EndMs)}}, true
	case "session.delegation.created":
		if ev.Delegation == nil {
			return nil, false
		}
		return []provider.RealtimeEvent{{
			Type:       provider.RealtimeDelegationCreated,
			ResponseID: ev.Delegation.ResponseID,
			Delegation: &provider.RealtimeDelegation{
				ID:         ev.Delegation.ID,
				Target:     ev.Delegation.Target,
				ResponseID: ev.Delegation.ResponseID,
				Offset:     time.Duration(ev.OffsetMs) * time.Millisecond,
			},
		}}, true
	case "response.event":
		return c.translateBackend(ev.Event)
	case "session.usage.updated", "session.closed":
		if ev.Usage == nil {
			return nil, true
		}
		usage := provider.RealtimeEvent{Type: provider.RealtimeUsageUpdated, UsageDuration: time.Duration(ev.Usage.Seconds * float64(time.Second))}
		meta := map[string]any{}
		if ev.ContextWindow != nil {
			meta["contextWindowUsageRatio"] = ev.ContextWindow.UsageRatio
		}
		if ev.Reason != "" {
			meta["closeReason"] = ev.Reason
		}
		if len(meta) > 0 {
			usage.ProviderMetadata = map[string]any{"openai": meta}
		}
		return []provider.RealtimeEvent{usage}, true
	case "error":
		return []provider.RealtimeEvent{{Type: provider.RealtimeError, Error: liveError(ev.Error)}}, true
	}
	return nil, false
}

// translateBackend reads a nested Responses event: completed function calls
// become tool calls; backend failures become error events.
func (c *liveConn) translateBackend(raw json.RawMessage) ([]provider.RealtimeEvent, bool) {
	var ev liveBackendEvent
	if len(raw) == 0 || json.Unmarshal(raw, &ev) != nil {
		return nil, false
	}
	switch ev.Type {
	case "response.output_item.done":
		if ev.Item == nil || ev.Item.Type != "function_call" {
			return nil, false
		}
		c.pendingMu.Lock()
		c.pending[ev.Item.CallID] = true
		c.pendingMu.Unlock()
		input := json.RawMessage(ev.Item.Arguments)
		if len(input) == 0 || !json.Valid(input) {
			input = json.RawMessage(`{}`)
		}
		return []provider.RealtimeEvent{{
			Type:       provider.RealtimeToolCallEvent,
			ToolCallID: ev.Item.CallID,
			ToolCall:   &provider.RealtimeToolCall{ID: ev.Item.CallID, Name: ev.Item.Name, Input: input},
		}}, true
	case "response.failed":
		var e *liveWireError
		if ev.Response != nil {
			e = ev.Response.Error
		}
		return []provider.RealtimeEvent{{Type: provider.RealtimeError, Error: fmt.Errorf("openai: live backend response failed: %w", liveError(e))}}, true
	case "error":
		e := ev.Error
		if e == nil {
			e = &liveWireError{Code: ev.Code, Message: ev.Message}
		}
		return []provider.RealtimeEvent{{Type: provider.RealtimeError, Error: fmt.Errorf("openai: live backend: %w", liveError(e))}}, true
	}
	return nil, false
}

func liveMs(ms int64) time.Duration { return time.Duration(ms) * time.Millisecond }
