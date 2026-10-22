package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sync"

	"github.com/zendev-sh/goai/internal/jsonmerge"
	"github.com/zendev-sh/goai/internal/openairealtime"
	"github.com/zendev-sh/goai/internal/realtimews"
	"github.com/zendev-sh/goai/provider"
)

var (
	_ provider.RealtimeModel        = (*translationModel)(nil)
	_ provider.RealtimeCapableModel = (*translationModel)(nil)
	_ provider.RealtimeConnection   = (*translationConn)(nil)
)

// translationInputFormat is the only input the translation API takes.
var translationInputFormat = provider.AudioFormat{Encoding: provider.AudioPCM16, SampleRate: 24_000, Channels: 1}

// Translation creates an OpenAI realtime translation model, such as
// "gpt-realtime-translate": it hears speech and speaks it in the target
// language, with optional source and translated transcripts. Set the target
// language with TranslationProviderOptions. Close sends session.close and
// waits for session.closed so the last translated audio is delivered.
//
// Experimental: see provider.RealtimeModel.
func Translation(modelID string, opts ...Option) provider.RealtimeModel {
	return &translationModel{id: modelID, opts: resolveConnectOptions(opts)}
}

// TranslationConfig holds the translation session settings. Language is the
// target language, such as "es"; NoiseReduction is "near_field" or "far_field".
type TranslationConfig struct {
	Language       string
	NoiseReduction string
}

// TranslationProviderOptions converts cfg to the ProviderOptions of a
// translation session.
func TranslationProviderOptions(cfg TranslationConfig) map[string]any {
	audio := map[string]any{}
	if cfg.Language != "" {
		audio["output"] = map[string]any{"language": cfg.Language}
	}
	if cfg.NoiseReduction != "" {
		audio["input"] = map[string]any{"noise_reduction": map[string]any{"type": cfg.NoiseReduction}}
	}
	if len(audio) == 0 {
		return map[string]any{}
	}
	return map[string]any{"audio": audio}
}

type translationModel struct {
	id   string
	opts options
}

func (m *translationModel) ModelID() string { return m.id }

// RealtimeCapabilities describes the translation API: 24 kHz mono PCM16 in;
// each output audio event carries its own format.
func (m *translationModel) RealtimeCapabilities() provider.RealtimeCapabilities {
	return provider.RealtimeCapabilities{
		InputAudio:           true,
		OutputAudio:          true,
		InputTranscription:   true,
		OutputTranscription:  true,
		RuntimeSessionUpdate: true,
		RawEvents:            true,
		SupportedInputAudio:  []provider.AudioFormat{translationInputFormat},
	}
}

// OpenRealtime dials the translation endpoint, waits for session.created, and
// applies the session settings, if any, before returning.
func (m *translationModel) OpenRealtime(ctx context.Context, params provider.RealtimeParams) (provider.RealtimeConnection, error) {
	if err := params.Validate(); err != nil {
		return nil, err
	}
	session, err := buildTranslationSession(params)
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
	endpoint, err := liveURL(m.opts.baseURL, "/realtime/translations")
	if err != nil {
		return nil, err
	}
	endpoint += "?model=" + url.QueryEscape(m.id)

	header := http.Header{}
	for k, v := range m.opts.headers {
		header.Set(k, v)
	}
	header.Set("Authorization", "Bearer "+token)
	ws, resp, err := realtimews.Dial(ctx, endpoint, realtimews.Options{HTTPClient: m.opts.httpClient, Headers: header, ReadLimit: liveReadLimit})
	if err != nil {
		return nil, liveDialError(err, resp)
	}

	connCtx, cancel := context.WithCancel(ctx)
	c := &translationConn{
		ws: ws, ctx: connCtx, cancel: cancel,
		events:    make(chan provider.RealtimeEvent, liveEventBuffer),
		done:      make(chan struct{}),
		closing:   make(chan struct{}),
		closed:    make(chan struct{}),
		inputTr:   params.InputTranscription.Enabled,
		outputTr:  params.OutputTranscription.Enabled,
		rawEvents: params.IncludeRawEvents,
	}
	sessionID, err := c.open(ctx, session)
	if err != nil {
		c.abort()
		return nil, err
	}
	c.setup = append(c.setup, provider.RealtimeEvent{Type: provider.RealtimeSessionReady, SessionID: sessionID})
	go c.readLoop()
	return c, nil
}

// buildTranslationSession maps the params to session.update; an empty map
// keeps the server defaults.
func buildTranslationSession(params provider.RealtimeParams) (map[string]any, error) {
	if params.System != "" || params.Voice != "" || len(params.Tools) > 0 || params.ToolChoice != "" || params.MaxOutputTokens > 0 {
		return nil, fmt.Errorf("%w: a translation session has no instructions, voice, or tools", provider.ErrRealtimeUnsupported)
	}
	if td := params.TurnDetection; td.Mode != "" || td.DisableAutoResponse || td.DisableInterruption || len(td.ProviderOptions) > 0 {
		return nil, fmt.Errorf("%w: translation streams continuously; it has no turns", provider.ErrRealtimeUnsupported)
	}
	if !params.InputAudio.IsZero() && params.InputAudio != translationInputFormat {
		return nil, fmt.Errorf("%w: translation input is 24 kHz mono PCM16", provider.ErrUnsupportedAudioFormat)
	}
	if !params.OutputAudio.IsZero() {
		return nil, fmt.Errorf("%w: the translation output format is chosen by the server", provider.ErrUnsupportedAudioFormat)
	}
	out := params.OutputTranscription
	if params.InputTranscription.Language != "" || params.InputTranscription.Prompt != "" ||
		out.Language != "" || out.Prompt != "" || len(out.ProviderOptions) > 0 {
		return nil, fmt.Errorf("%w: translation transcripts take only a model", provider.ErrRealtimeUnsupported)
	}
	session := map[string]any{}
	if params.InputTranscription.Enabled {
		model := openairealtime.DefaultTranscriptionModel
		if m, ok := params.InputTranscription.ProviderOptions["model"].(string); ok && m != "" {
			model = m
		}
		session["audio"] = map[string]any{"input": map[string]any{"transcription": map[string]any{"model": model}}}
	}
	return jsonmerge.Deep(session, params.ProviderOptions), nil
}

// translationEvent is the envelope of every translation server event.
type translationEvent struct {
	Type       string `json:"type"`
	Delta      string `json:"delta"`
	ElapsedMs  int64  `json:"elapsed_ms"`
	SampleRate int    `json:"sample_rate"`
	Channels   int    `json:"channels"`
	Format     string `json:"format"`
	Session    *struct {
		ID string `json:"id"`
	} `json:"session"`
	Error *liveWireError `json:"error"`
}

// translationConn is one translation WebSocket session. One reader goroutine
// translates server events; writes may come from any goroutine.
type translationConn struct {
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

	inputTr, outputTr, rawEvents bool

	// Reader goroutine only.
	setup []provider.RealtimeEvent
	seq   int64
}

func (c *translationConn) Events() <-chan provider.RealtimeEvent { return c.events }

func (c *translationConn) Err() error {
	c.errMu.Lock()
	defer c.errMu.Unlock()
	return c.err
}

func (c *translationConn) setErr(err error) {
	if err == nil {
		return
	}
	c.errMu.Lock()
	if c.err == nil {
		c.err = err
	}
	c.errMu.Unlock()
}

// open waits for session.created and, with settings, for session.updated.
func (c *translationConn) open(ctx context.Context, session map[string]any) (string, error) {
	var id string
	configured := len(session) == 0
	for {
		ev, data, err := c.read(ctx)
		if err != nil {
			return "", err
		}
		switch ev.Type {
		case "session.created":
			if ev.Session != nil {
				id = ev.Session.ID
			}
			if c.rawEvents {
				c.setup = append(c.setup, liveRawEvent(data))
			}
			if configured {
				return id, nil
			}
			if err := c.write(ctx, map[string]any{"type": "session.update", "session": session}); err != nil {
				return "", fmt.Errorf("openai: sending translation session.update: %w", err)
			}
		case "session.updated":
			if c.rawEvents {
				c.setup = append(c.setup, liveRawEvent(data))
			}
			return id, nil
		case "error":
			return "", translationError(ev.Error)
		default:
			c.setup = append(c.setup, c.translateMessage(ev, data)...)
		}
	}
}

func (c *translationConn) read(ctx context.Context) (translationEvent, []byte, error) {
	_, data, err := c.ws.Read(ctx)
	if err != nil {
		code, reason, ok := realtimews.CloseStatus(err)
		switch {
		case !ok:
			return translationEvent{}, nil, fmt.Errorf("openai: translation connection lost: %w", err)
		case code == realtimews.StatusNormalClosure:
			return translationEvent{}, nil, errLiveClosedNormally
		default:
			return translationEvent{}, nil, fmt.Errorf("openai: translation session closed with status %d: %s", code, reason)
		}
	}
	var ev translationEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return translationEvent{}, nil, fmt.Errorf("openai: decoding translation event: %w", err)
	}
	return ev, data, nil
}

func (c *translationConn) write(ctx context.Context, ev map[string]any) error {
	data, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	return c.ws.Write(ctx, realtimews.MessageText, data)
}

// Close sends session.close and waits up to liveCloseTimeout for
// session.closed, so translated audio still in flight arrives, then closes the
// WebSocket. Idempotent.
func (c *translationConn) Close() error {
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

func (c *translationConn) abort() {
	c.closeOnce.Do(func() {
		_ = c.ws.CloseNow()
		c.cancel()
		close(c.events)
		close(c.done)
	})
}

func (c *translationConn) Send(ctx context.Context, event provider.RealtimeClientEvent) error {
	if c.ctx.Err() != nil {
		return provider.ErrRealtimeSessionClosed
	}
	select {
	case <-c.closing:
		return provider.ErrRealtimeSessionClosed
	default:
	}
	var data []byte
	switch event.Type {
	case provider.RealtimeRawEvent:
		if !realtimews.IsJSONObject(event.Raw) {
			return fmt.Errorf("%w: raw event must be a JSON object", provider.ErrInvalidRealtimeConfig)
		}
		data = event.Raw
	case provider.RealtimeInputAudioAppend:
		if len(event.Audio) == 0 {
			return fmt.Errorf("%w: empty audio frame", provider.ErrInvalidRealtimeConfig)
		}
		data, _ = json.Marshal(map[string]any{"type": "session.input_audio_buffer.append", "audio": base64.StdEncoding.EncodeToString(event.Audio)})
	case provider.RealtimeSessionUpdateEvent:
		update := event.Update
		if update == nil || len(update.ProviderOptions) == 0 {
			return fmt.Errorf("%w: a translation update takes ProviderOptions such as TranslationProviderOptions", provider.ErrInvalidRealtimeConfig)
		}
		if update.System != nil || update.Tools != nil {
			return fmt.Errorf("%w: a translation session has no instructions or tools", provider.ErrRealtimeUnsupported)
		}
		data, _ = json.Marshal(map[string]any{"type": "session.update", "session": update.ProviderOptions})
	default:
		return fmt.Errorf("%w: realtime translation does not support %s", provider.ErrRealtimeUnsupported, event.Type)
	}
	if err := c.ws.Write(ctx, realtimews.MessageText, data); err != nil {
		if c.ctx.Err() != nil {
			return provider.ErrRealtimeSessionClosed
		}
		return fmt.Errorf("openai: sending translation event: %w", err)
	}
	return nil
}

func (c *translationConn) readLoop() {
	defer close(c.done)
	defer close(c.events)
	for _, event := range c.setup {
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
			if !c.emit(event) {
				return
			}
		}
		if ev.Type == "session.closed" {
			c.closedOnce.Do(func() { close(c.closed) })
		}
	}
}

// emit queues an event in order. After Close it only uses free queue space,
// so a consumer that stopped reading cannot block the close handshake.
func (c *translationConn) emit(event provider.RealtimeEvent) bool {
	c.seq++
	event.Sequence = c.seq
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

func (c *translationConn) translateMessage(ev translationEvent, data []byte) []provider.RealtimeEvent {
	events, known := c.translate(ev)
	if !c.rawEvents && known {
		return events
	}
	return append([]provider.RealtimeEvent{liveRawEvent(data)}, events...)
}

// translate maps one server event; elapsed_ms is stream alignment metadata.
func (c *translationConn) translate(ev translationEvent) ([]provider.RealtimeEvent, bool) {
	var meta map[string]any
	if ev.ElapsedMs > 0 {
		meta = map[string]any{"openai": map[string]any{"elapsedMs": ev.ElapsedMs}}
	}
	switch ev.Type {
	case "session.output_audio.delta":
		audio, err := base64.StdEncoding.DecodeString(ev.Delta)
		if err != nil {
			return []provider.RealtimeEvent{{Type: provider.RealtimeError, Error: fmt.Errorf("openai: decoding translated audio: %w", err)}}, true
		}
		format := provider.AudioFormat{Encoding: provider.AudioPCM16, SampleRate: ev.SampleRate, Channels: ev.Channels}
		if ev.Format != "" && ev.Format != "pcm16" {
			format.Encoding = provider.AudioEncoding(ev.Format)
		}
		return []provider.RealtimeEvent{{Type: provider.RealtimeOutputAudio, Audio: audio, AudioFormat: format, ProviderMetadata: meta}}, true
	case "session.input_transcript.delta":
		if !c.inputTr {
			return nil, true
		}
		return []provider.RealtimeEvent{{Type: provider.RealtimeInputTranscript, Text: ev.Delta, ProviderMetadata: meta}}, true
	case "session.output_transcript.delta":
		if !c.outputTr {
			return nil, true
		}
		return []provider.RealtimeEvent{{Type: provider.RealtimeOutputTranscript, Text: ev.Delta, ProviderMetadata: meta}}, true
	case "session.closed":
		return nil, true
	case "error":
		return []provider.RealtimeEvent{{Type: provider.RealtimeError, Error: translationError(ev.Error)}}, true
	}
	return nil, false
}

func translationError(e *liveWireError) error {
	if e == nil {
		return errors.New("openai: translation error")
	}
	return fmt.Errorf("openai: translation %s (%s): %s", cmpString(e.Code, e.Type), e.Type, e.Message)
}
