// Package openairealtime implements the OpenAI Realtime API (GA interface) over
// WebSocket. It is shared by provider/openai and provider/azure, which only
// differ in endpoint and credentials.
package openairealtime

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
	"github.com/zendev-sh/goai/internal/realtimews"
	"github.com/zendev-sh/goai/provider"
)

var (
	_ provider.RealtimeModel        = (*model)(nil)
	_ provider.RealtimeCapableModel = (*model)(nil)
	_ provider.RealtimeConnection   = (*conn)(nil)
)

const (
	readLimit    = 16 << 20
	eventBuffer  = 64
	maxErrorBody = 1024

	// DefaultTranscriptionModel transcribes input audio when the caller does not
	// pick a model through TranscriptionConfig.ProviderOptions["model"].
	DefaultTranscriptionModel = "gpt-live-transcribe"
)

var (
	pcm24k = provider.AudioFormat{Encoding: provider.AudioPCM16, SampleRate: 24_000, Channels: 1}
	pcmu8k = provider.AudioFormat{Encoding: provider.AudioG711U, SampleRate: 8_000, Channels: 1}
	pcma8k = provider.AudioFormat{Encoding: provider.AudioG711A, SampleRate: 8_000, Channels: 1}
)

// errClosedNormally reports a clean close from the server; it is not an error
// once the session is running.
var errClosedNormally = errors.New("realtime connection closed")

// Config is the backend-specific part of a Realtime connection.
type Config struct {
	// Provider names the backend in errors, e.g. "openai" or "azure".
	Provider string
	// ModelID is the model, or the deployment name on Azure.
	ModelID string
	// BaseURL is the HTTP(S) API base; the session dials BaseURL + "/realtime".
	BaseURL    string
	Headers    map[string]string
	HTTPClient *http.Client
	// Authorize sets the credential headers of the WebSocket handshake.
	Authorize func(ctx context.Context, header http.Header) error
}

// New returns a realtime model speaking the OpenAI Realtime protocol.
func New(cfg Config) provider.RealtimeModel { return &model{cfg: cfg} }

type model struct{ cfg Config }

func (m *model) ModelID() string { return m.cfg.ModelID }

// RealtimeCapabilities describes the GA Realtime protocol over WebSocket.
func (m *model) RealtimeCapabilities() provider.RealtimeCapabilities {
	formats := []provider.AudioFormat{pcm24k, pcmu8k, pcma8k}
	return provider.RealtimeCapabilities{
		InputAudio:              true,
		OutputAudio:             true,
		InputTranscription:      true,
		OutputTranscription:     true,
		TextInput:               true,
		ImageInput:              true,
		ProviderVAD:             true,
		ManualTurnControl:       true,
		ExplicitResponseRequest: true,
		BargeIn:                 true,
		OutputAudioTruncation:   true,
		IdleTimeout:             true,
		FunctionCalling:         true,
		BlockingToolCalls:       true,
		ParallelToolCalls:       true,
		RuntimeSessionUpdate:    true,
		ConversationAppend:      true,
		UsageUpdates:            true,
		RateLimitUpdates:        true,
		RawEvents:               true,
		SupportedInputAudio:     formats,
		SupportedOutputAudio:    append([]provider.AudioFormat(nil), formats...),
	}
}

// OpenRealtime dials the Realtime endpoint, applies the session configuration
// with session.update, and returns once the server confirms it.
func (m *model) OpenRealtime(ctx context.Context, params provider.RealtimeParams) (provider.RealtimeConnection, error) {
	if err := params.Validate(); err != nil {
		return nil, err
	}
	session, outFormat, err := buildSession(params)
	if err != nil {
		return nil, err
	}
	// Caller headers first, credentials last, so a caller header cannot override them.
	header := http.Header{}
	for k, v := range m.cfg.Headers {
		header.Set(k, v)
	}
	if m.cfg.Authorize == nil {
		return nil, fmt.Errorf("%s: no credentials configured", m.cfg.Provider)
	}
	if err := m.cfg.Authorize(ctx, header); err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}
	endpoint, err := realtimeURL(m.cfg.BaseURL, m.cfg.ModelID)
	if err != nil {
		return nil, err
	}

	ws, resp, err := realtimews.Dial(ctx, endpoint, realtimews.Options{
		HTTPClient: m.cfg.HTTPClient,
		Headers:    header,
		ReadLimit:  readLimit,
	})
	if err != nil {
		return nil, dialError(m.cfg.Provider, err, resp)
	}

	connCtx, cancel := context.WithCancel(ctx)
	c := &conn{
		provider:     m.cfg.Provider,
		ws:           ws,
		ctx:          connCtx,
		cancel:       cancel,
		events:       make(chan provider.RealtimeEvent, eventBuffer),
		done:         make(chan struct{}),
		closing:      make(chan struct{}),
		outFormat:    outFormat,
		autoResponse: !params.TurnDetection.DisableAutoResponse,
		outputTr:     params.OutputTranscription.Enabled,
		rawEvents:    params.IncludeRawEvents,
		responses:    newSequencer(),
		audioMs:      make(map[string]int64),
	}
	if err := c.configure(ctx, session); err != nil {
		c.abort()
		return nil, err
	}
	c.setup = append(c.setup, provider.RealtimeEvent{Type: provider.RealtimeSessionReady, SessionID: c.sessionID})
	go c.readLoop()
	return c, nil
}

func realtimeURL(base, modelID string) (string, error) {
	u, err := url.Parse(strings.TrimRight(base, "/"))
	if err != nil {
		return "", fmt.Errorf("invalid realtime base URL: %w", err)
	}
	switch u.Scheme {
	case "https", "wss":
		u.Scheme = "wss"
	case "http", "ws":
		u.Scheme = "ws"
	default:
		return "", fmt.Errorf("unsupported realtime base URL scheme %q", u.Scheme)
	}
	u.Path += "/realtime"
	q := u.Query()
	q.Set("model", modelID)
	u.RawQuery = q.Encode()
	return u.String(), nil
}

func dialError(providerID string, err error, resp *http.Response) error {
	if resp == nil || resp.StatusCode == http.StatusSwitchingProtocols {
		return fmt.Errorf("%s: connecting to realtime api: %w", providerID, err)
	}
	var body []byte
	if resp.Body != nil {
		body, _ = io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
		_ = resp.Body.Close()
	}
	return goai.ParseHTTPErrorWithHeaders(providerID, resp.StatusCode, body, resp.Header)
}

// buildSession maps RealtimeParams to the GA session object sent in
// session.update, and returns the negotiated output audio format.
func buildSession(params provider.RealtimeParams) (map[string]any, provider.AudioFormat, error) {
	if params.Resumption != nil {
		return nil, provider.AudioFormat{}, fmt.Errorf("%w: openai realtime has no session resumption", provider.ErrRealtimeUnsupported)
	}
	inFormat, err := wireFormat(params.InputAudio, "input")
	if err != nil {
		return nil, provider.AudioFormat{}, err
	}
	outFormat, err := wireFormat(params.OutputAudio, "output")
	if err != nil {
		return nil, provider.AudioFormat{}, err
	}
	out := params.OutputAudio
	if out.IsZero() {
		out = pcm24k
	}

	input := map[string]any{"format": inFormat}
	if params.InputTranscription.Enabled {
		input["transcription"] = buildTranscription(params.InputTranscription)
	}
	turn, err := buildTurnDetection(params.TurnDetection)
	if err != nil {
		return nil, provider.AudioFormat{}, err
	}
	if turn.set {
		input["turn_detection"] = turn.value
	}

	output := map[string]any{"format": outFormat}
	if params.Voice != "" {
		output["voice"] = params.Voice
	}

	session := map[string]any{
		"type":              "realtime",
		"output_modalities": []string{"audio"},
		"audio":             map[string]any{"input": input, "output": output},
	}
	if params.System != "" {
		session["instructions"] = params.System
	}
	if params.MaxOutputTokens > 0 {
		session["max_output_tokens"] = params.MaxOutputTokens
	}
	if len(params.Tools) > 0 {
		tools, err := BuildTools(params.Tools)
		if err != nil {
			return nil, provider.AudioFormat{}, err
		}
		session["tools"] = tools
	}
	if choice := ToolChoice(params.ToolChoice); choice != nil {
		session["tool_choice"] = choice
	}
	session, err = mergeProviderOptions(session, params.ProviderOptions)
	if err != nil {
		return nil, provider.AudioFormat{}, err
	}
	return session, out, nil
}

func wireFormat(f provider.AudioFormat, side string) (map[string]any, error) {
	switch {
	case f.IsZero() || f == pcm24k:
		return map[string]any{"type": "audio/pcm", "rate": 24_000}, nil
	case f == pcmu8k:
		return map[string]any{"type": "audio/pcmu"}, nil
	case f == pcma8k:
		return map[string]any{"type": "audio/pcma"}, nil
	default:
		return nil, fmt.Errorf("%w: realtime %s must be 24 kHz mono PCM16 or 8 kHz mono G.711", provider.ErrUnsupportedAudioFormat, side)
	}
}

func buildTranscription(cfg provider.TranscriptionConfig) map[string]any {
	tc := map[string]any{"model": DefaultTranscriptionModel}
	if cfg.Language != "" {
		tc["language"] = cfg.Language
	}
	if cfg.Prompt != "" {
		tc["prompt"] = cfg.Prompt
	}
	for k, v := range cfg.ProviderOptions {
		tc[k] = v
	}
	return tc
}

type turnDetection struct {
	set   bool
	value any // nil marshals to JSON null: manual turns
}

func buildTurnDetection(td provider.TurnDetection) (turnDetection, error) {
	switch td.Mode {
	case provider.TurnModeManual:
		return turnDetection{set: true, value: nil}, nil
	case provider.TurnModeClientVAD:
		return turnDetection{}, fmt.Errorf("%w: realtime has no client activity signals; use manual turns", provider.ErrRealtimeUnsupported)
	}
	if !td.DisableAutoResponse && !td.DisableInterruption && len(td.ProviderOptions) == 0 {
		return turnDetection{}, nil // provider default: server VAD
	}
	value := map[string]any{"type": "server_vad"}
	for k, v := range td.ProviderOptions {
		value[k] = v
	}
	if td.DisableAutoResponse {
		value["create_response"] = false
	}
	if td.DisableInterruption {
		value["interrupt_response"] = false
	}
	return turnDetection{set: true, value: value}, nil
}

// BuildTools maps tool definitions to Realtime tools: functions, and MCP
// servers declared as provider-defined "mcp" tools.
func BuildTools(defs []provider.ToolDefinition) ([]map[string]any, error) {
	tools := make([]map[string]any, 0, len(defs))
	for _, def := range defs {
		if def.ProviderDefinedType == "mcp" {
			tool := map[string]any{"type": "mcp"}
			for k, v := range def.ProviderDefinedOptions {
				tool[k] = v
			}
			tools = append(tools, tool)
			continue
		}
		if def.ProviderDefinedType != "" {
			return nil, fmt.Errorf("%w: provider-defined tool %s in a realtime session", provider.ErrRealtimeUnsupported, def.Name)
		}
		params := json.RawMessage(`{"type":"object","properties":{}}`)
		if len(def.InputSchema) > 0 {
			if !json.Valid(def.InputSchema) {
				return nil, fmt.Errorf("%w: invalid schema for tool %s", provider.ErrInvalidRealtimeConfig, def.Name)
			}
			params = def.InputSchema
		}
		tools = append(tools, map[string]any{
			"type":        "function",
			"name":        def.Name,
			"description": def.Description,
			"parameters":  params,
		})
	}
	return tools, nil
}

// ToolChoice maps a tool choice to its Realtime value; nil keeps the default.
func ToolChoice(choice string) any {
	switch choice {
	case "":
		return nil
	case "auto", "none", "required":
		return choice
	default:
		return map[string]any{"type": "function", "name": choice}
	}
}

// mergeProviderOptions overlays flat ProviderOptions on the session; nested
// objects merge at every level. Fields the SDK derives cannot be overridden.
func mergeProviderOptions(session, opts map[string]any) (map[string]any, error) {
	for key := range opts {
		switch key {
		case "type", "model", "instructions", "tools":
			return nil, fmt.Errorf("%w: %s is derived from the realtime params", provider.ErrInvalidRealtimeConfig, key)
		}
	}
	return jsonmerge.Deep(session, opts), nil
}

// conn is one Realtime WebSocket session. One reader goroutine translates
// server events; writes may come from any goroutine.
type conn struct {
	provider string
	ws       *realtimews.Conn
	ctx      context.Context
	cancel   context.CancelFunc
	events   chan provider.RealtimeEvent
	done     chan struct{}
	closing  chan struct{}

	closeOnce sync.Once

	errMu sync.Mutex
	err   error

	outFormat    provider.AudioFormat
	autoResponse bool
	outputTr     bool
	rawEvents    bool

	responses *sequencer

	audioMu sync.Mutex
	audioMs map[string]int64 // received output audio per item, for truncation

	// Reader goroutine only. setup holds events produced while configuring,
	// delivered before anything the reader reads.
	setup     []provider.RealtimeEvent
	sessionID string
	seq       int64
}

func (c *conn) Events() <-chan provider.RealtimeEvent { return c.events }

func (c *conn) Err() error {
	c.errMu.Lock()
	defer c.errMu.Unlock()
	return c.err
}

func (c *conn) setErr(err error) {
	if err == nil {
		return
	}
	c.errMu.Lock()
	if c.err == nil {
		c.err = err
	}
	c.errMu.Unlock()
}

// Close performs the close handshake and waits for the reader. Idempotent.
func (c *conn) Close() error {
	c.closeOnce.Do(func() {
		close(c.closing)
		_ = c.ws.Close()
		c.cancel()
	})
	<-c.done
	return nil
}

func (c *conn) abort() {
	c.closeOnce.Do(func() {
		_ = c.ws.CloseNow()
		c.cancel()
		close(c.events)
		close(c.done)
	})
}

// configure sends session.update and waits for session.updated. Server errors
// during setup fail the open.
func (c *conn) configure(ctx context.Context, session map[string]any) error {
	if err := c.write(ctx, clientEvent{"type": "session.update", "session": session}); err != nil {
		return fmt.Errorf("%s: sending session.update: %w", c.provider, err)
	}
	for {
		ev, data, err := c.read(ctx)
		if err != nil {
			if errors.Is(err, errClosedNormally) {
				return fmt.Errorf("%s: realtime connection closed during setup", c.provider)
			}
			return err
		}
		switch ev.Type {
		case "session.updated":
			if c.rawEvents {
				c.setup = append(c.setup, rawEvent(data))
			}
			return nil
		case "error":
			return c.serverError(ev.Error)
		}
		c.setup = append(c.setup, c.translateMessage(ev, data)...)
	}
}

func (c *conn) read(ctx context.Context) (serverEvent, []byte, error) {
	_, data, err := c.ws.Read(ctx)
	if err != nil {
		if closeErr := c.closeError(err); closeErr != nil {
			return serverEvent{}, nil, closeErr
		}
		return serverEvent{}, nil, errClosedNormally
	}
	var ev serverEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return serverEvent{}, nil, fmt.Errorf("%s: decoding realtime event: %w", c.provider, err)
	}
	return ev, data, nil
}

// translateMessage adds the raw event to its normalized events: always with
// IncludeRawEvents, otherwise when the adapter does not know its type.
func (c *conn) translateMessage(ev serverEvent, data []byte) []provider.RealtimeEvent {
	events, known := c.translate(ev)
	if !c.rawEvents && known {
		return events
	}
	return append([]provider.RealtimeEvent{rawEvent(data)}, events...)
}

func rawEvent(data []byte) provider.RealtimeEvent {
	return provider.RealtimeEvent{Type: provider.RealtimeProviderEvent, Raw: append(json.RawMessage(nil), data...)}
}

func (c *conn) closeError(err error) error {
	code, reason, ok := realtimews.CloseStatus(err)
	if !ok {
		return fmt.Errorf("%s: realtime connection lost: %w", c.provider, err)
	}
	if code == realtimews.StatusNormalClosure {
		return nil
	}
	return fmt.Errorf("%s: realtime session closed with status %d: %s", c.provider, code, reason)
}

func (c *conn) serverError(e *wireError) error {
	if e == nil {
		return fmt.Errorf("%s: realtime error", c.provider)
	}
	return fmt.Errorf("%s: realtime %s (%s): %s", c.provider, cmpOr(e.Code, e.Type), e.Type, e.Message)
}

func cmpOr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func (c *conn) write(ctx context.Context, ev clientEvent) error {
	data, err := marshalEvent(ev)
	if err != nil {
		return err
	}
	return c.ws.Write(ctx, realtimews.MessageText, data)
}

func (c *conn) Send(ctx context.Context, event provider.RealtimeClientEvent) error {
	if c.ctx.Err() != nil {
		return provider.ErrRealtimeSessionClosed
	}
	if event.Type == provider.RealtimeRawEvent {
		if !realtimews.IsJSONObject(event.Raw) {
			return fmt.Errorf("%w: raw event must be a JSON object", provider.ErrInvalidRealtimeConfig)
		}
		return c.sendError(c.ws.Write(ctx, realtimews.MessageText, event.Raw))
	}
	msgs, response, err := c.encode(event)
	if err != nil {
		return err
	}
	// Content goes out before any response request is registered: otherwise the
	// reader could send a queued response.create ahead of it on response.done.
	for _, msg := range msgs {
		if err := c.writeEvent(ctx, msg); err != nil {
			return err
		}
	}
	if event.Type == provider.RealtimeToolResultEvent {
		c.responses.toolOutput(event.ToolResult.ToolCallID)
	}
	if response != nil {
		c.responses.request(response.params)
	}
	return c.dispatch(ctx)
}

// dispatch sends the next queued response.create when the slot is free.
func (c *conn) dispatch(ctx context.Context) error {
	ev, ok := c.responses.next()
	if !ok {
		return nil
	}
	if err := c.writeEvent(ctx, ev); err != nil {
		c.responses.sendFailed(ev["event_id"].(string))
		return err
	}
	return nil
}

func (c *conn) writeEvent(ctx context.Context, msg clientEvent) error {
	return c.sendError(c.write(ctx, msg))
}

// sendError maps a failed write to the error a sender sees.
func (c *conn) sendError(err error) error {
	if err == nil {
		return nil
	}
	if c.ctx.Err() != nil {
		return provider.ErrRealtimeSessionClosed
	}
	return fmt.Errorf("%s: sending realtime event: %w", c.provider, err)
}

// encodeInterrupt cancels the active response, if any, and truncates the
// played item to the audio that actually reached it. Like the official Agents
// SDK, nothing is cancelled without an active response and truncation is
// clamped to the audio received for that item.
func (c *conn) encodeInterrupt(cur *provider.PlaybackCursor) []clientEvent {
	var msgs []clientEvent
	if c.responses.active() {
		msgs = append(msgs, clientEvent{"type": "response.cancel"})
	}
	if cur == nil || cur.ItemID == "" {
		return msgs
	}
	c.audioMu.Lock()
	received, known := c.audioMs[cur.ItemID]
	c.audioMu.Unlock()
	if !known {
		return msgs
	}
	played := min(max(cur.PlayedFor.Milliseconds(), 0), received)
	return append(msgs, clientEvent{
		"type":          "conversation.item.truncate",
		"item_id":       cur.ItemID,
		"content_index": 0,
		"audio_end_ms":  played,
	})
}

func (c *conn) readLoop() {
	defer close(c.done)
	defer close(c.events)
	if !c.emit(c.setup) {
		return
	}
	c.setup = nil
	for {
		ev, data, err := c.read(c.ctx)
		if err != nil {
			if c.ctx.Err() == nil {
				if !errors.Is(err, errClosedNormally) {
					c.setErr(err)
				}
				_ = c.ws.CloseNow()
			}
			return
		}
		if !c.emit(c.translateMessage(ev, data)) {
			return
		}
		if err := c.dispatch(c.ctx); err != nil && c.ctx.Err() == nil {
			c.setErr(fmt.Errorf("%s: creating the next response: %w", c.provider, err))
		}
	}
}

// emit delivers events in order and reports false once the connection ends.
// After Close they are dropped so the reader can finish the close handshake.
func (c *conn) emit(events []provider.RealtimeEvent) bool {
	for _, event := range events {
		c.seq++
		event.Sequence = c.seq
		select {
		case c.events <- event:
		case <-c.closing:
		case <-c.ctx.Done():
			return false
		}
	}
	return true
}

// translate maps one server event to common events and reports whether the
// adapter knows its type. Reader goroutine only.
func (c *conn) translate(ev serverEvent) ([]provider.RealtimeEvent, bool) {
	one := func(e provider.RealtimeEvent) ([]provider.RealtimeEvent, bool) {
		return []provider.RealtimeEvent{e}, true
	}
	switch ev.Type {
	case "session.created":
		// The session ID travels on RealtimeSessionReady, sent once configured.
		if ev.Session != nil {
			c.sessionID = ev.Session.ID
		}
		return nil, true
	case "input_audio_buffer.speech_started":
		return one(provider.RealtimeEvent{Type: provider.RealtimeInputSpeechStarted, ItemID: ev.ItemID, AudioStart: msDuration(ev.AudioStartMs)})
	case "input_audio_buffer.speech_stopped":
		return one(provider.RealtimeEvent{Type: provider.RealtimeInputSpeechStopped, ItemID: ev.ItemID, AudioEnd: msDuration(ev.AudioEndMs)})
	case "conversation.item.input_audio_transcription.delta":
		return one(provider.RealtimeEvent{Type: provider.RealtimeInputTranscript, ItemID: ev.ItemID, Text: ev.Delta,
			ProviderMetadata: c.metadata("logprobs", ev.Logprobs)})
	case "conversation.item.input_audio_transcription.completed":
		return one(provider.RealtimeEvent{Type: provider.RealtimeInputTranscript, ItemID: ev.ItemID, Text: ev.Transcript, IsFinal: true,
			ProviderMetadata: c.metadata("logprobs", ev.Logprobs, "usage", ev.Usage)})
	case "conversation.item.input_audio_transcription.segment":
		return one(provider.RealtimeEvent{Type: provider.RealtimeInputTranscript, ItemID: ev.ItemID, Text: ev.Text, Speaker: ev.Speaker,
			AudioStart: secondsToDuration(ev.Start), AudioEnd: secondsToDuration(ev.End),
			ProviderMetadata: c.metadata("segmentId", ev.ID)})
	case "conversation.item.input_audio_transcription.failed":
		return one(provider.RealtimeEvent{Type: provider.RealtimeError, ItemID: ev.ItemID, Error: c.serverError(ev.Error)})
	case "response.created":
		var id string
		inConversation := true
		var meta map[string]any
		if r := ev.Response; r != nil {
			id = r.ID
			inConversation = string(r.ConversationID) != "null"
			meta = c.metadata("metadata", r.Metadata, "conversationId", r.ConversationID)
		}
		c.responses.created(id, inConversation)
		return one(provider.RealtimeEvent{Type: provider.RealtimeResponseStarted, ResponseID: id, ProviderMetadata: meta})
	case "response.output_audio.delta":
		audio, err := base64.StdEncoding.DecodeString(ev.Delta)
		if err != nil {
			return one(provider.RealtimeEvent{Type: provider.RealtimeError, Error: fmt.Errorf("%s: decoding output audio: %w", c.provider, err)})
		}
		c.audioMu.Lock()
		c.audioMs[ev.ItemID] += audioDuration(len(audio), c.outFormat)
		c.audioMu.Unlock()
		return one(provider.RealtimeEvent{Type: provider.RealtimeOutputAudio, ResponseID: ev.ResponseID, ItemID: ev.ItemID, Audio: audio, AudioFormat: c.outFormat})
	case "response.output_audio_transcript.delta":
		if !c.outputTr {
			return nil, true
		}
		return one(provider.RealtimeEvent{Type: provider.RealtimeOutputTranscript, ResponseID: ev.ResponseID, ItemID: ev.ItemID, Text: ev.Delta})
	case "response.output_audio_transcript.done":
		if !c.outputTr {
			return nil, true
		}
		return one(provider.RealtimeEvent{Type: provider.RealtimeOutputTranscript, ResponseID: ev.ResponseID, ItemID: ev.ItemID, Text: ev.Transcript, IsFinal: true})
	case "response.output_text.delta":
		return one(provider.RealtimeEvent{Type: provider.RealtimeTextDelta, ResponseID: ev.ResponseID, ItemID: ev.ItemID, Text: ev.Delta})
	case "response.output_text.done":
		return one(provider.RealtimeEvent{Type: provider.RealtimeTextDelta, ResponseID: ev.ResponseID, ItemID: ev.ItemID, Text: ev.Text, IsFinal: true})
	case "response.function_call_arguments.done":
		c.responses.toolCall(ev.CallID, ev.ResponseID)
		input := json.RawMessage(ev.Arguments)
		if len(input) == 0 || !json.Valid(input) {
			input = json.RawMessage(`{}`)
		}
		return one(provider.RealtimeEvent{
			Type:       provider.RealtimeToolCallEvent,
			ResponseID: ev.ResponseID,
			ItemID:     ev.ItemID,
			ToolCallID: ev.CallID,
			ToolCall:   &provider.RealtimeToolCall{ID: ev.CallID, Name: ev.Name, Input: input},
		})
	case "response.done":
		id := ""
		if ev.Response != nil {
			id = ev.Response.ID
		}
		c.responses.done(id)
		return c.responseDone(ev.Response), true
	case "rate_limits.updated":
		limits := make([]provider.RealtimeRateLimit, 0, len(ev.RateLimits))
		for _, rl := range ev.RateLimits {
			limits = append(limits, provider.RealtimeRateLimit{
				Name:       rl.Name,
				Limit:      rl.Limit,
				Remaining:  rl.Remaining,
				ResetAfter: secondsToDuration(rl.ResetSeconds),
			})
		}
		return one(provider.RealtimeEvent{Type: provider.RealtimeRateLimitsUpdated, RateLimits: limits})
	case "error":
		// A rejected response.create is dropped, not retried (see sequencer).
		c.responses.rejected(ev.Error)
		return one(provider.RealtimeEvent{Type: provider.RealtimeError, Error: c.serverError(ev.Error)})
	}
	return nil, false
}

// metadata builds ProviderMetadata under the provider's key from key/value
// pairs, leaving out empty values. Raw JSON values are decoded.
func (c *conn) metadata(kv ...any) map[string]any {
	fields := map[string]any{}
	for i := 0; i+1 < len(kv); i += 2 {
		key, _ := kv[i].(string)
		switch v := kv[i+1].(type) {
		case json.RawMessage:
			var decoded any
			if len(v) > 0 && json.Unmarshal(v, &decoded) == nil && decoded != nil {
				fields[key] = decoded
			}
		case string:
			if v != "" {
				fields[key] = v
			}
		case map[string]any:
			if len(v) > 0 {
				fields[key] = v
			}
		}
	}
	if len(fields) == 0 {
		return nil
	}
	return map[string]any{c.provider: fields}
}

func msDuration(ms int64) time.Duration { return time.Duration(ms) * time.Millisecond }

// audioDuration returns the milliseconds of audio in n bytes of format f.
func audioDuration(n int, f provider.AudioFormat) int64 {
	bytesPerSample := 1 // G.711
	if f.Encoding == provider.AudioPCM16 {
		bytesPerSample = 2
	}
	perMs := f.SampleRate * f.Channels * bytesPerSample / 1000
	if perMs == 0 {
		return 0
	}
	return int64(n / perMs)
}

func (c *conn) responseDone(resp *wireResponse) []provider.RealtimeEvent {
	if resp == nil {
		return []provider.RealtimeEvent{{Type: provider.RealtimeResponseFinished, FinishReason: provider.FinishStop}}
	}
	finished := provider.RealtimeEvent{
		Type:             provider.RealtimeResponseFinished,
		ResponseID:       resp.ID,
		FinishReason:     finishReason(resp),
		ProviderMetadata: c.metadata("metadata", resp.Metadata, "conversationId", resp.ConversationID),
	}
	events := []provider.RealtimeEvent{finished}
	if resp.Status == "failed" {
		var detail *wireError
		if resp.StatusDetails != nil {
			detail = resp.StatusDetails.Error
		}
		events = append(events, provider.RealtimeEvent{Type: provider.RealtimeError, ResponseID: resp.ID, Error: c.serverError(detail)})
	}
	if u := resp.Usage; u != nil {
		cached := u.cachedTokens()
		events[0].Usage = provider.Usage{
			InputTokens:     max(u.InputTokens-cached, 0),
			CacheReadTokens: cached,
			OutputTokens:    u.OutputTokens,
			TotalTokens:     u.TotalTokens,
		}
		events = append(events, provider.RealtimeEvent{
			Type: provider.RealtimeUsageUpdated, ResponseID: resp.ID, Usage: events[0].Usage,
			ProviderMetadata: c.metadata("inputTokenDetails", u.InputTokenDetails, "outputTokenDetails", u.OutputTokenDetails),
		})
	}
	return events
}

func secondsToDuration(s float64) time.Duration {
	return time.Duration(s * float64(time.Second))
}

func finishReason(resp *wireResponse) provider.FinishReason {
	reason := ""
	if resp.StatusDetails != nil {
		reason = resp.StatusDetails.Reason
	}
	switch resp.Status {
	case "cancelled":
		return provider.FinishInterrupted
	case "failed":
		return provider.FinishError
	case "incomplete":
		if reason == "content_filter" {
			return provider.FinishContentFilter
		}
		return provider.FinishLength
	}
	for _, item := range resp.Output {
		if item.Type == "function_call" {
			return provider.FinishToolCalls
		}
	}
	return provider.FinishStop
}
