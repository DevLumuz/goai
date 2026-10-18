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
		ProviderVAD:             true,
		ManualTurnControl:       true,
		ExplicitResponseRequest: true,
		BargeIn:                 true,
		OutputAudioTruncation:   true,
		IdleTimeout:             true,
		FunctionCalling:         true,
		BlockingToolCalls:       true,
		ParallelToolCalls:       true,
		UsageUpdates:            true,
		RateLimitUpdates:        true,
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
		pending:      make(map[string]bool),
		audioMs:      make(map[string]int64),
	}
	if err := c.configure(ctx, session); err != nil {
		c.abort()
		return nil, err
	}
	c.events <- provider.RealtimeEvent{Type: provider.RealtimeSessionReady}
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
		tools, err := buildTools(params.Tools)
		if err != nil {
			return nil, provider.AudioFormat{}, err
		}
		session["tools"] = tools
	}
	if choice := buildToolChoice(params.ToolChoice); choice != nil {
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

func buildTools(defs []provider.ToolDefinition) ([]map[string]any, error) {
	tools := make([]map[string]any, 0, len(defs))
	for _, def := range defs {
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

func buildToolChoice(choice string) any {
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

	// stateMu guards response sequencing, the tool loop, and audio lengths.
	// Like the official Agents SDK, response.create is never sent while a
	// response is in progress: requests made meanwhile coalesce into one.
	stateMu        sync.Mutex
	ongoing        bool // between response.created and response.done
	createInFlight bool // response.create sent, response.created not yet seen
	wantCreate     bool // a response was requested while another was active
	pending        map[string]bool
	toolTurn       bool
	respDone       bool
	audioMs        map[string]int64 // received output audio per item, for truncation

	seq int64 // reader goroutine only
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
		ev, err := c.read(ctx)
		if err != nil {
			if errors.Is(err, errClosedNormally) {
				return fmt.Errorf("%s: realtime connection closed during setup", c.provider)
			}
			return err
		}
		switch ev.Type {
		case "session.updated":
			return nil
		case "error":
			return c.serverError(ev.Error)
		}
	}
}

func (c *conn) read(ctx context.Context) (serverEvent, error) {
	_, data, err := c.ws.Read(ctx)
	if err != nil {
		if closeErr := c.closeError(err); closeErr != nil {
			return serverEvent{}, closeErr
		}
		return serverEvent{}, errClosedNormally
	}
	var ev serverEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return serverEvent{}, fmt.Errorf("%s: decoding realtime event: %w", c.provider, err)
	}
	return ev, nil
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
	msgs, respond, err := c.encode(event)
	if err != nil {
		return err
	}
	// Content goes out before any response request is registered: otherwise the
	// reader could send a coalesced response.create ahead of it on response.done.
	for _, msg := range msgs {
		if err := c.writeEvent(ctx, msg); err != nil {
			return err
		}
	}
	if event.Type == provider.RealtimeToolResultEvent {
		respond = c.toolOutputSent(event.ToolResult.ToolCallID)
	}
	if respond && c.requestResponse() {
		return c.writeEvent(ctx, clientEvent{"type": "response.create"})
	}
	return nil
}

func (c *conn) writeEvent(ctx context.Context, msg clientEvent) error {
	if err := c.write(ctx, msg); err != nil {
		if c.ctx.Err() != nil {
			return provider.ErrRealtimeSessionClosed
		}
		return fmt.Errorf("%s: sending realtime event: %w", c.provider, err)
	}
	return nil
}

// requestResponse reports whether response.create can be sent now. Otherwise
// the request is kept and sent when the active response finishes.
func (c *conn) requestResponse() bool {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()
	if c.ongoing || c.createInFlight {
		c.wantCreate = true
		return false
	}
	c.createInFlight = true
	return true
}

// encode maps a client event to wire events. respond reports whether the
// event also asks for a model response, which goes through requestResponse.
func (c *conn) encode(event provider.RealtimeClientEvent) (msgs []clientEvent, respond bool, err error) {
	switch event.Type {
	case provider.RealtimeInputAudioAppend:
		if len(event.Audio) == 0 {
			return nil, false, fmt.Errorf("%w: empty audio frame", provider.ErrInvalidRealtimeConfig)
		}
		return []clientEvent{{"type": "input_audio_buffer.append", "audio": base64.StdEncoding.EncodeToString(event.Audio)}}, false, nil
	case provider.RealtimeInputAudioCommit:
		return []clientEvent{{"type": "input_audio_buffer.commit"}}, false, nil
	case provider.RealtimeResponseCreate:
		return nil, true, nil
	case provider.RealtimeTextInput:
		return []clientEvent{{"type": "conversation.item.create", "item": map[string]any{
			"type":    "message",
			"role":    "user",
			"content": []map[string]any{{"type": "input_text", "text": event.Text}},
		}}}, c.autoResponse, nil
	case provider.RealtimeToolResultEvent:
		result := event.ToolResult
		if result == nil || result.ToolCallID == "" {
			return nil, false, fmt.Errorf("%w: tool result needs a tool call ID", provider.ErrInvalidRealtimeConfig)
		}
		return []clientEvent{{"type": "conversation.item.create", "item": map[string]any{
			"type":    "function_call_output",
			"call_id": result.ToolCallID,
			"output":  string(result.Output),
		}}}, false, nil
	case provider.RealtimeResponseCancel:
		return c.encodeInterrupt(event.Cursor), false, nil
	default:
		return nil, false, fmt.Errorf("%w: openai realtime does not support %s", provider.ErrRealtimeUnsupported, event.Type)
	}
}

// encodeInterrupt cancels the active response, if any, and truncates the
// played item to the audio that actually reached it. Like the official Agents
// SDK, nothing is cancelled without an active response and truncation is
// clamped to the audio received for that item.
func (c *conn) encodeInterrupt(cur *provider.PlaybackCursor) []clientEvent {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()
	var msgs []clientEvent
	if c.ongoing {
		msgs = append(msgs, clientEvent{"type": "response.cancel"})
	}
	c.wantCreate = false
	if cur == nil || cur.ItemID == "" {
		return msgs
	}
	received, known := c.audioMs[cur.ItemID]
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

// toolOutputSent records a tool output and reports whether it was the last
// one the finished response was waiting for.
func (c *conn) toolOutputSent(callID string) bool {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()
	delete(c.pending, callID)
	return c.continueToolTurn()
}

// continueToolTurn must be called with stateMu held.
func (c *conn) continueToolTurn() bool {
	if c.toolTurn && c.respDone && len(c.pending) == 0 {
		c.toolTurn = false
		return true
	}
	return false
}

func (c *conn) readLoop() {
	defer close(c.done)
	defer close(c.events)
	for {
		ev, err := c.read(c.ctx)
		if err != nil {
			if c.ctx.Err() == nil {
				if !errors.Is(err, errClosedNormally) {
					c.setErr(err)
				}
				_ = c.ws.CloseNow()
			}
			return
		}
		for _, event := range c.translate(ev) {
			c.seq++
			event.Sequence = c.seq
			select {
			case c.events <- event:
			case <-c.closing:
			case <-c.ctx.Done():
				return
			}
		}
		if ev.Type == "response.done" && c.finishResponse() {
			if err := c.write(c.ctx, clientEvent{"type": "response.create"}); err != nil && c.ctx.Err() == nil {
				c.setErr(fmt.Errorf("%s: creating the next response: %w", c.provider, err))
			}
		}
	}
}

// finishResponse closes the active response and reports whether a
// response.create is due: a coalesced request or a completed tool turn.
func (c *conn) finishResponse() bool {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()
	c.ongoing = false
	c.respDone = true
	if c.continueToolTurn() {
		c.wantCreate = true
	}
	if c.wantCreate && !c.createInFlight {
		c.wantCreate = false
		c.createInFlight = true
		return true
	}
	return false
}

// translate maps one server event to common events. Reader goroutine only.
func (c *conn) translate(ev serverEvent) []provider.RealtimeEvent {
	switch ev.Type {
	case "input_audio_buffer.speech_started":
		return []provider.RealtimeEvent{{Type: provider.RealtimeInputSpeechStarted, ItemID: ev.ItemID}}
	case "input_audio_buffer.speech_stopped":
		return []provider.RealtimeEvent{{Type: provider.RealtimeInputSpeechStopped, ItemID: ev.ItemID}}
	case "conversation.item.input_audio_transcription.delta":
		return []provider.RealtimeEvent{{Type: provider.RealtimeInputTranscript, ItemID: ev.ItemID, Text: ev.Delta}}
	case "conversation.item.input_audio_transcription.completed":
		return []provider.RealtimeEvent{{Type: provider.RealtimeInputTranscript, ItemID: ev.ItemID, Text: ev.Transcript, IsFinal: true}}
	case "conversation.item.input_audio_transcription.failed":
		return []provider.RealtimeEvent{{Type: provider.RealtimeError, ItemID: ev.ItemID, Error: c.serverError(ev.Error)}}
	case "response.created":
		c.stateMu.Lock()
		c.ongoing, c.createInFlight = true, false
		c.respDone, c.toolTurn = false, false
		clear(c.pending)
		c.stateMu.Unlock()
		id := ""
		if ev.Response != nil {
			id = ev.Response.ID
		}
		return []provider.RealtimeEvent{{Type: provider.RealtimeResponseStarted, ResponseID: id}}
	case "response.output_audio.delta":
		audio, err := base64.StdEncoding.DecodeString(ev.Delta)
		if err != nil {
			return []provider.RealtimeEvent{{Type: provider.RealtimeError, Error: fmt.Errorf("%s: decoding output audio: %w", c.provider, err)}}
		}
		c.stateMu.Lock()
		c.audioMs[ev.ItemID] += audioDuration(len(audio), c.outFormat)
		c.stateMu.Unlock()
		return []provider.RealtimeEvent{{Type: provider.RealtimeOutputAudio, ResponseID: ev.ResponseID, ItemID: ev.ItemID, Audio: audio, AudioFormat: c.outFormat}}
	case "response.output_audio_transcript.delta":
		if !c.outputTr {
			return nil
		}
		return []provider.RealtimeEvent{{Type: provider.RealtimeOutputTranscript, ResponseID: ev.ResponseID, ItemID: ev.ItemID, Text: ev.Delta}}
	case "response.output_audio_transcript.done":
		if !c.outputTr {
			return nil
		}
		return []provider.RealtimeEvent{{Type: provider.RealtimeOutputTranscript, ResponseID: ev.ResponseID, ItemID: ev.ItemID, Text: ev.Transcript, IsFinal: true}}
	case "response.output_text.delta":
		return []provider.RealtimeEvent{{Type: provider.RealtimeTextDelta, ResponseID: ev.ResponseID, ItemID: ev.ItemID, Text: ev.Delta}}
	case "response.function_call_arguments.done":
		c.stateMu.Lock()
		c.pending[ev.CallID] = true
		c.toolTurn = true
		c.stateMu.Unlock()
		input := json.RawMessage(ev.Arguments)
		if len(input) == 0 || !json.Valid(input) {
			input = json.RawMessage(`{}`)
		}
		return []provider.RealtimeEvent{{
			Type:       provider.RealtimeToolCallEvent,
			ResponseID: ev.ResponseID,
			ItemID:     ev.ItemID,
			ToolCallID: ev.CallID,
			ToolCall:   &provider.RealtimeToolCall{ID: ev.CallID, Name: ev.Name, Input: input},
		}}
	case "response.done":
		return c.responseDone(ev.Response)
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
		return []provider.RealtimeEvent{{Type: provider.RealtimeRateLimitsUpdated, RateLimits: limits}}
	case "error":
		if ev.Error != nil && ev.Error.Code == "conversation_already_has_active_response" {
			// Our request raced a server-created response: retry after it ends.
			c.stateMu.Lock()
			c.createInFlight, c.wantCreate = false, true
			c.stateMu.Unlock()
		}
		return []provider.RealtimeEvent{{Type: provider.RealtimeError, Error: c.serverError(ev.Error)}}
	}
	return nil
}

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
		Type:         provider.RealtimeResponseFinished,
		ResponseID:   resp.ID,
		FinishReason: finishReason(resp),
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
		cached := 0
		if u.InputTokenDetails != nil {
			cached = u.InputTokenDetails.CachedTokens
		}
		events[0].Usage = provider.Usage{
			InputTokens:     max(u.InputTokens-cached, 0),
			CacheReadTokens: cached,
			OutputTokens:    u.OutputTokens,
			TotalTokens:     u.TotalTokens,
		}
		events = append(events, provider.RealtimeEvent{Type: provider.RealtimeUsageUpdated, ResponseID: resp.ID, Usage: events[0].Usage})
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
