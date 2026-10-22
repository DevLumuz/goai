package provider

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

var (
	// ErrRealtimeUnsupported is returned when a realtime model does not support
	// the requested operation or configuration.
	ErrRealtimeUnsupported = errors.New("goai: realtime operation not supported")

	// ErrRealtimeSessionClosed is returned when an operation targets a closed
	// realtime connection.
	ErrRealtimeSessionClosed = errors.New("goai: realtime session closed")

	// ErrUnsupportedAudioFormat is returned when an audio format is invalid or
	// unsupported by the selected realtime model.
	ErrUnsupportedAudioFormat = errors.New("goai: unsupported realtime audio format")

	// ErrRealtimeBackpressure is returned when a bounded realtime queue cannot
	// accept more data without accumulating stale audio or events.
	ErrRealtimeBackpressure = errors.New("goai: realtime backpressure limit reached")

	// ErrInvalidRealtimeConfig is returned when common realtime configuration is
	// internally inconsistent before provider-specific validation is applied.
	ErrInvalidRealtimeConfig = errors.New("goai: invalid realtime configuration")
)

// AudioEncoding identifies the encoding of realtime audio bytes.
type AudioEncoding string

const (
	// AudioPCM16 is signed 16-bit little-endian PCM.
	AudioPCM16 AudioEncoding = "pcm_s16le"
	// AudioG711U is G.711 mu-law audio.
	AudioG711U AudioEncoding = "g711_ulaw"
	// AudioG711A is G.711 A-law audio.
	AudioG711A AudioEncoding = "g711_alaw"
)

// AudioFormat describes the bytes sent to or received from a realtime model.
type AudioFormat struct {
	Encoding   AudioEncoding
	SampleRate int
	Channels   int
}

// IsZero reports whether no audio format was configured.
func (f AudioFormat) IsZero() bool {
	return f.Encoding == "" && f.SampleRate == 0 && f.Channels == 0
}

// Validate checks provider-independent audio format constraints. A provider may
// reject an otherwise valid format when its selected model does not support it.
func (f AudioFormat) Validate() error {
	if f.IsZero() {
		return nil
	}
	if f.Encoding == "" {
		return fmt.Errorf("%w: encoding is required", ErrUnsupportedAudioFormat)
	}
	if f.SampleRate <= 0 {
		return fmt.Errorf("%w: sample rate must be positive", ErrUnsupportedAudioFormat)
	}
	if f.Channels <= 0 {
		return fmt.Errorf("%w: channels must be positive", ErrUnsupportedAudioFormat)
	}
	return nil
}

// TurnMode selects who decides when an input turn begins and ends.
type TurnMode string

const (
	// TurnModeProviderVAD delegates voice activity detection to the provider.
	TurnModeProviderVAD TurnMode = "provider_vad"
	// TurnModeClientVAD lets the client signal speech activity explicitly.
	TurnModeClientVAD TurnMode = "client_vad"
	// TurnModeManual requires explicit input commit and response requests.
	TurnModeManual TurnMode = "manual"
)

// TurnDetection configures provider-independent turn behavior. ProviderOptions
// contains experimental provider-specific fields for the selected model.
//
// The zero value keeps the provider defaults: automatic responses at the end of
// a turn and barge-in when the user starts speaking.
type TurnDetection struct {
	Mode TurnMode
	// DisableAutoResponse stops the provider from responding when it detects
	// the end of a turn; the client calls for a response explicitly.
	DisableAutoResponse bool
	// DisableInterruption keeps the model speaking when user activity starts.
	DisableInterruption bool
	ProviderOptions     map[string]any
}

// TranscriptionConfig requests transcription for one side of the conversation.
// Model-specific fields belong in ProviderOptions until they have shared
// semantics across providers.
type TranscriptionConfig struct {
	Enabled         bool
	Language        string
	Prompt          string
	ProviderOptions map[string]any
}

// RealtimeParams contains the initial configuration of a realtime session.
// Common behavior is represented by typed fields. ProviderOptions is an escape
// hatch for provider-specific experimental settings and is interpreted only by
// the selected provider implementation.
type RealtimeParams struct {
	System              string
	Tools               []ToolDefinition
	ToolChoice          string
	InputAudio          AudioFormat
	OutputAudio         AudioFormat
	Voice               string
	TurnDetection       TurnDetection
	InputTranscription  TranscriptionConfig
	OutputTranscription TranscriptionConfig
	MaxOutputTokens     int
	// IncludeRawEvents delivers every provider message as a RealtimeProviderEvent
	// in addition to the normalized events. Without it, only messages and parts
	// the adapter does not map arrive that way.
	IncludeRawEvents bool
	ProviderOptions  map[string]any
}

// Validate checks provider-independent realtime configuration. Providers must
// additionally validate capabilities and model-specific restrictions.
func (p RealtimeParams) Validate() error {
	if err := p.InputAudio.Validate(); err != nil {
		return fmt.Errorf("%w: input audio: %w", ErrInvalidRealtimeConfig, err)
	}
	if err := p.OutputAudio.Validate(); err != nil {
		return fmt.Errorf("%w: output audio: %w", ErrInvalidRealtimeConfig, err)
	}
	switch p.TurnDetection.Mode {
	case "", TurnModeProviderVAD, TurnModeClientVAD, TurnModeManual:
	default:
		return fmt.Errorf("%w: unknown turn mode %q", ErrInvalidRealtimeConfig, p.TurnDetection.Mode)
	}
	if p.MaxOutputTokens < 0 {
		return fmt.Errorf("%w: max output tokens cannot be negative", ErrInvalidRealtimeConfig)
	}
	return nil
}

// FinishInterrupted reports a realtime response cut short by user activity or
// an explicit interruption rather than completed by the model.
const FinishInterrupted FinishReason = "interrupted"

// RealtimeClientEventType identifies a command sent to a realtime model.
type RealtimeClientEventType string

const (
	RealtimeInputAudioAppend   RealtimeClientEventType = "input_audio_append"
	RealtimeInputAudioCommit   RealtimeClientEventType = "input_audio_commit"
	RealtimeInputActivityStart RealtimeClientEventType = "input_activity_start"
	RealtimeInputActivityEnd   RealtimeClientEventType = "input_activity_end"
	RealtimeTextInput          RealtimeClientEventType = "text_input"
	RealtimeToolResultEvent    RealtimeClientEventType = "tool_result"
	RealtimeResponseCreate     RealtimeClientEventType = "response_create"
	RealtimeResponseCancel     RealtimeClientEventType = "response_cancel"
	// RealtimeContextAppend adds application context to a delegating voice model.
	RealtimeContextAppend RealtimeClientEventType = "context_append"
	// RealtimeImageInput sends an encoded image or video frame.
	RealtimeImageInput RealtimeClientEventType = "image_input"
	// RealtimeInputAudioEnd reports that the input audio stream paused or ended,
	// for example because the microphone was muted.
	RealtimeInputAudioEnd RealtimeClientEventType = "input_audio_end"
	// RealtimeConversationAppend adds messages to the conversation history.
	RealtimeConversationAppend RealtimeClientEventType = "conversation_append"
	// RealtimeSessionUpdateEvent changes the configuration of a live session.
	RealtimeSessionUpdateEvent RealtimeClientEventType = "session_update"
	// RealtimeRawEvent writes provider JSON as given. It bypasses the adapter's
	// state tracking, so the caller owns its consequences.
	RealtimeRawEvent RealtimeClientEventType = "raw"
)

// RealtimeMedia is an encoded image or video frame. The application captures
// and encodes it; adapters send and deliver the bytes as they are.
type RealtimeMedia struct {
	MIMEType string
	Data     []byte
}

// RealtimeConversation adds messages to the conversation history, for example
// to restore a previous conversation or inject text turns.
type RealtimeConversation struct {
	Messages []Message
	// RequestResponse asks the model to respond after the messages are added.
	RequestResponse bool
}

// RealtimeSessionUpdate changes a live session. Nil fields keep the current
// value; a non-nil empty Tools removes every tool.
type RealtimeSessionUpdate struct {
	System          *string
	Tools           []ToolDefinition
	ProviderOptions map[string]any
}

// RealtimeContextKind selects how a delegating voice model uses appended context.
type RealtimeContextKind string

const (
	// RealtimeContextCommentary is said aloud; the model paraphrases it.
	RealtimeContextCommentary RealtimeContextKind = "commentary"
	// RealtimeContextThinking informs later replies without being spoken now.
	RealtimeContextThinking RealtimeContextKind = "thinking"
	// RealtimeContextInstructions adds instructions to the live conversation and
	// may interrupt speech in progress.
	RealtimeContextInstructions RealtimeContextKind = "instructions"
)

// RealtimeContext is application content for a delegating voice model. An
// empty DelegationID marks session-wide context instead of a task result.
type RealtimeContext struct {
	Kind         RealtimeContextKind
	DelegationID string
	Text         string
}

// RealtimeDelegation describes work a voice model hands to a backend. Target is
// "client" when the application runs it, or the provider's backend otherwise.
type RealtimeDelegation struct {
	ID         string
	Target     string
	ResponseID string
	// Offset is the position in the session audio where the delegation started.
	Offset time.Duration
}

// PlaybackCursor identifies how much generated audio was actually played. It
// allows providers that support truncation to keep conversation state aligned
// with what the user heard.
type PlaybackCursor struct {
	ResponseID string
	ItemID     string
	PlayedFor  time.Duration
}

// RealtimeToolResult is a structured result sent for a realtime tool call.
type RealtimeToolResult struct {
	ToolCallID string
	Output     json.RawMessage
	IsError    bool
	// Partial marks an intermediate result: more results for the same call follow.
	Partial bool
	// Media carries images or other encoded content returned with the result.
	Media           []RealtimeMedia
	ProviderOptions map[string]any
}

// RealtimeClientEvent is a normalized command sent through RealtimeConnection.
// Only fields relevant to Type are used.
type RealtimeClientEvent struct {
	Type         RealtimeClientEventType
	Audio        []byte
	Text         string
	ToolResult   *RealtimeToolResult
	Cursor       *PlaybackCursor
	Context      *RealtimeContext
	Media        *RealtimeMedia
	Conversation *RealtimeConversation
	Update       *RealtimeSessionUpdate
	Raw          json.RawMessage
	// ProviderOptions are provider-specific fields the adapter merges into the
	// message it writes, such as the parameters of a requested response.
	ProviderOptions map[string]any
}

// RealtimeEventType identifies an event emitted by a realtime model.
type RealtimeEventType string

const (
	RealtimeSessionReady       RealtimeEventType = "session_ready"
	RealtimeInputSpeechStarted RealtimeEventType = "input_speech_started"
	RealtimeInputSpeechStopped RealtimeEventType = "input_speech_stopped"
	RealtimeInputTranscript    RealtimeEventType = "input_transcript"
	RealtimeResponseStarted    RealtimeEventType = "response_started"
	RealtimeOutputAudio        RealtimeEventType = "output_audio"
	RealtimeOutputTranscript   RealtimeEventType = "output_transcript"
	RealtimeTextDelta          RealtimeEventType = "text_delta"
	RealtimeToolCallEvent      RealtimeEventType = "tool_call"
	RealtimeToolCallCancelled  RealtimeEventType = "tool_call_cancelled"
	RealtimeResponseFinished   RealtimeEventType = "response_finished"
	RealtimeUsageUpdated       RealtimeEventType = "usage_updated"
	RealtimeRateLimitsUpdated  RealtimeEventType = "rate_limits_updated"
	RealtimeContextCompacted   RealtimeEventType = "context_compacted"
	RealtimeContextTruncated   RealtimeEventType = "context_truncated"
	RealtimeSessionResumeState RealtimeEventType = "session_resume_state"
	RealtimeSessionGoAway      RealtimeEventType = "session_go_away"
	RealtimeDelegationCreated  RealtimeEventType = "delegation_created"
	RealtimeError              RealtimeEventType = "error"
	// RealtimeReasoningDelta carries model reasoning text, not spoken output.
	RealtimeReasoningDelta RealtimeEventType = "reasoning_delta"
	// RealtimeSources reports the sources a response was grounded on.
	RealtimeSources RealtimeEventType = "sources"
	// RealtimeOutputMedia carries an image or video frame generated by the model.
	RealtimeOutputMedia RealtimeEventType = "output_media"
	// RealtimeWaitingForInput reports that the model is waiting for the user.
	RealtimeWaitingForInput RealtimeEventType = "waiting_for_input"
	// RealtimeGenerationComplete reports that the model finished generating the
	// response; playback of buffered audio may still be in progress.
	RealtimeGenerationComplete RealtimeEventType = "generation_complete"
	// RealtimeProviderEvent carries a provider message as raw JSON.
	RealtimeProviderEvent RealtimeEventType = "provider_event"
)

// RealtimeToolCall describes a tool invocation requested by the model.
type RealtimeToolCall struct {
	ID    string
	Name  string
	Input json.RawMessage
}

// RealtimeRateLimit describes one provider-reported realtime quota.
type RealtimeRateLimit struct {
	Name       string
	Limit      int
	Remaining  int
	ResetAfter time.Duration
}

// RealtimeEvent is a normalized event emitted by a realtime connection. Fields
// not applicable to Type remain at their zero values.
//
// For transcript events, Text holds an incremental delta while IsFinal is false
// and the complete transcript of the turn when IsFinal is true. An interim
// transcript is a revisable hypothesis that later events replace.
type RealtimeEvent struct {
	Type        RealtimeEventType
	SessionID   string
	ResponseID  string
	ItemID      string
	Sequence    int64
	Audio       []byte
	AudioFormat AudioFormat
	Text        string
	IsFinal     bool
	IsInterim   bool
	// AudioStart and AudioEnd locate speech or a transcript in the input audio.
	AudioStart time.Duration
	AudioEnd   time.Duration
	Language   string
	Speaker    string
	Media      *RealtimeMedia
	Sources    []Source
	ToolCall   *RealtimeToolCall
	ToolCallID string
	// Pending marks a finished response whose interaction continues, such as a
	// function still running in the background.
	Pending      bool
	FinishReason FinishReason
	Usage        Usage
	// UsageDuration is the cumulative billed session time for providers that
	// charge by time; each update replaces the previous one.
	UsageDuration time.Duration
	Delegation    *RealtimeDelegation
	RateLimits    []RealtimeRateLimit
	ResumeToken   string
	TimeToClose   time.Duration
	Error         error
	// Raw is the provider JSON of a RealtimeProviderEvent.
	Raw              json.RawMessage
	ProviderMetadata map[string]any
}

// RealtimeCapabilities declares the behavior available for one realtime model
// and configuration. Zero values mean unsupported or unknown.
type RealtimeCapabilities struct {
	InputAudio              bool
	OutputAudio             bool
	InputTranscription      bool
	OutputTranscription     bool
	TextInput               bool
	ImageInput              bool
	ProviderVAD             bool
	ClientVAD               bool
	ManualTurnControl       bool
	ExplicitResponseRequest bool
	BargeIn                 bool
	OutputAudioTruncation   bool
	IdleTimeout             bool
	FunctionCalling         bool
	BlockingToolCalls       bool
	NonBlockingToolCalls    bool
	ParallelToolCalls       bool
	ToolCallCancellation    bool
	RuntimeSessionUpdate    bool
	SessionResumption       bool
	ContextCompression      bool
	UsageUpdates            bool
	RateLimitUpdates        bool
	// Delegation reports a voice model that hands reasoning and tools to a
	// backend and accepts RealtimeContextAppend results.
	Delegation         bool
	OutputMedia        bool
	ConversationAppend bool
	InputAudioEnd      bool
	PartialToolResults bool
	// RawEvents reports support for RealtimeRawEvent and IncludeRawEvents.
	RawEvents            bool
	SupportedInputAudio  []AudioFormat
	SupportedOutputAudio []AudioFormat
}

// Clone returns capabilities whose slice fields do not alias the receiver.
func (c RealtimeCapabilities) Clone() RealtimeCapabilities {
	c.SupportedInputAudio = append([]AudioFormat(nil), c.SupportedInputAudio...)
	c.SupportedOutputAudio = append([]AudioFormat(nil), c.SupportedOutputAudio...)
	return c
}
