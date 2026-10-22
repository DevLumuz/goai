package openai

import (
	"encoding/json"
	"maps"
	"time"

	"github.com/zendev-sh/goai/internal/openairealtime"
	"github.com/zendev-sh/goai/provider"
)

// RealtimeConfig holds the OpenAI Realtime session settings without a common
// realtime field, as in the official SDK's session request; zero values keep
// the defaults. Use it with goai.WithRealtimeProviderOptions(RealtimeProviderOptions(cfg)).
type RealtimeConfig struct {
	// OutputModalities is ["audio"] (the default) or ["text"].
	OutputModalities []string
	// VoiceID selects a custom voice by ID instead of a named voice.
	VoiceID string
	Speed   *float64
	// NoiseReduction is "near_field" or "far_field".
	NoiseReduction string
	TurnDetection  *RealtimeTurnDetection
	Transcription  *RealtimeTranscription
	// Include adds optional output, such as "item.input_audio_transcription.logprobs".
	Include           []string
	Prompt            *RealtimePrompt
	Tracing           *RealtimeTracing
	Truncation        *RealtimeTruncation
	ReasoningEffort   string
	ParallelToolCalls *bool
}

// RealtimeTurnDetection tunes voice activity detection. Type is "server_vad"
// (the default) or "semantic_vad"; Eagerness applies to semantic_vad and the
// thresholds and durations to server_vad.
type RealtimeTurnDetection struct {
	Type            string
	Eagerness       string
	Threshold       *float64
	PrefixPadding   time.Duration
	SilenceDuration time.Duration
	// IdleTimeout asks the model to respond after this much user silence.
	IdleTimeout time.Duration
}

// RealtimeTranscription tunes input transcription; setting it enables it.
type RealtimeTranscription struct {
	Model     string
	Keywords  []string
	Languages []string
	// Delay trades latency for accuracy, for example "low" or "high".
	Delay string
}

// RealtimePrompt references a stored prompt template.
type RealtimePrompt struct {
	ID        string
	Version   string
	Variables map[string]any
}

// RealtimeTracing enables tracing; zero values let the server choose names.
type RealtimeTracing struct {
	WorkflowName string
	GroupID      string
	Metadata     map[string]any
}

// RealtimeTruncation controls how the conversation is shortened when it
// exceeds the context window. Disabled fails instead; RetentionRatio keeps
// that fraction of the conversation.
type RealtimeTruncation struct {
	Disabled               bool
	RetentionRatio         float64
	PostInstructionsTokens int
}

// RealtimeProviderOptions converts cfg to the ProviderOptions of an OpenAI or
// Azure OpenAI Realtime session, in Realtime API field names.
func RealtimeProviderOptions(cfg RealtimeConfig) map[string]any {
	session := map[string]any{}
	if len(cfg.OutputModalities) > 0 {
		session["output_modalities"] = append([]string(nil), cfg.OutputModalities...)
	}
	input := map[string]any{}
	if cfg.NoiseReduction != "" {
		input["noise_reduction"] = map[string]any{"type": cfg.NoiseReduction}
	}
	if td := cfg.TurnDetection; td != nil {
		detection := map[string]any{"type": cmpString(td.Type, "server_vad")}
		putNonEmpty(detection, "eagerness", td.Eagerness)
		if td.Threshold != nil {
			detection["threshold"] = *td.Threshold
		}
		putMillis(detection, "prefix_padding_ms", td.PrefixPadding)
		putMillis(detection, "silence_duration_ms", td.SilenceDuration)
		putMillis(detection, "idle_timeout_ms", td.IdleTimeout)
		input["turn_detection"] = detection
	}
	if tr := cfg.Transcription; tr != nil {
		transcription := map[string]any{}
		putNonEmpty(transcription, "model", tr.Model)
		putNonEmpty(transcription, "delay", tr.Delay)
		if len(tr.Keywords) > 0 {
			transcription["keywords"] = append([]string(nil), tr.Keywords...)
		}
		if len(tr.Languages) > 0 {
			transcription["languages"] = append([]string(nil), tr.Languages...)
		}
		input["transcription"] = transcription
	}
	output := map[string]any{}
	if cfg.VoiceID != "" {
		output["voice"] = map[string]any{"id": cfg.VoiceID}
	}
	if cfg.Speed != nil {
		output["speed"] = *cfg.Speed
	}
	audio := map[string]any{}
	if len(input) > 0 {
		audio["input"] = input
	}
	if len(output) > 0 {
		audio["output"] = output
	}
	if len(audio) > 0 {
		session["audio"] = audio
	}
	if len(cfg.Include) > 0 {
		session["include"] = append([]string(nil), cfg.Include...)
	}
	if cfg.Prompt != nil {
		session["prompt"] = cfg.Prompt.wire()
	}
	if t := cfg.Tracing; t != nil {
		if t.WorkflowName == "" && t.GroupID == "" && len(t.Metadata) == 0 {
			session["tracing"] = "auto"
		} else {
			tracing := map[string]any{}
			putNonEmpty(tracing, "workflow_name", t.WorkflowName)
			putNonEmpty(tracing, "group_id", t.GroupID)
			if len(t.Metadata) > 0 {
				tracing["metadata"] = maps.Clone(t.Metadata)
			}
			session["tracing"] = tracing
		}
	}
	if t := cfg.Truncation; t != nil {
		switch {
		case t.Disabled:
			session["truncation"] = "disabled"
		case t.RetentionRatio > 0:
			truncation := map[string]any{"type": "retention_ratio", "retention_ratio": t.RetentionRatio}
			if t.PostInstructionsTokens > 0 {
				truncation["token_limits"] = map[string]any{"post_instructions": t.PostInstructionsTokens}
			}
			session["truncation"] = truncation
		default:
			session["truncation"] = "auto"
		}
	}
	if cfg.ReasoningEffort != "" {
		session["reasoning"] = map[string]any{"effort": cfg.ReasoningEffort}
	}
	if cfg.ParallelToolCalls != nil {
		session["parallel_tool_calls"] = *cfg.ParallelToolCalls
	}
	return session
}

func (p *RealtimePrompt) wire() map[string]any {
	prompt := map[string]any{"id": p.ID}
	putNonEmpty(prompt, "version", p.Version)
	if len(p.Variables) > 0 {
		prompt["variables"] = maps.Clone(p.Variables)
	}
	return prompt
}

// RealtimeResponseOptions are the parameters of one requested response, for
// RealtimeSession.RequestResponseWith. They apply to that response only.
type RealtimeResponseOptions struct {
	// OutOfBand keeps the response out of the default conversation, for side
	// tasks such as classification while the conversation goes on.
	OutOfBand bool
	// InputItemIDs and Input replace the conversation as the response context:
	// references to existing items first, then new messages.
	InputItemIDs     []string
	Input            []provider.Message
	Instructions     string
	Tools            []provider.ToolDefinition
	ToolChoice       string
	OutputModalities []string
	MaxOutputTokens  int
	Voice            string
	ReasoningEffort  string
	Prompt           *RealtimePrompt
	// Metadata is returned with the response events, to match responses to
	// requests.
	Metadata map[string]string
}

// RealtimeResponse converts o to the ProviderOptions of RequestResponseWith.
func RealtimeResponse(o RealtimeResponseOptions) (map[string]any, error) {
	response := map[string]any{}
	if o.OutOfBand {
		response["conversation"] = "none"
	}
	if len(o.InputItemIDs) > 0 || len(o.Input) > 0 {
		input := make([]map[string]any, 0, len(o.InputItemIDs)+len(o.Input))
		for _, id := range o.InputItemIDs {
			input = append(input, map[string]any{"type": "item_reference", "id": id})
		}
		if len(o.Input) > 0 {
			items, err := openairealtime.ConversationItems(o.Input)
			if err != nil {
				return nil, err
			}
			input = append(input, items...)
		}
		response["input"] = input
	}
	putNonEmpty(response, "instructions", o.Instructions)
	if o.Tools != nil {
		tools, err := openairealtime.BuildTools(o.Tools)
		if err != nil {
			return nil, err
		}
		response["tools"] = tools
	}
	if choice := openairealtime.ToolChoice(o.ToolChoice); choice != nil {
		response["tool_choice"] = choice
	}
	if len(o.OutputModalities) > 0 {
		response["output_modalities"] = append([]string(nil), o.OutputModalities...)
	}
	if o.MaxOutputTokens > 0 {
		response["max_output_tokens"] = o.MaxOutputTokens
	}
	if o.Voice != "" {
		response["audio"] = map[string]any{"output": map[string]any{"voice": o.Voice}}
	}
	if o.ReasoningEffort != "" {
		response["reasoning"] = map[string]any{"effort": o.ReasoningEffort}
	}
	if o.Prompt != nil {
		response["prompt"] = o.Prompt.wire()
	}
	if len(o.Metadata) > 0 {
		response["metadata"] = maps.Clone(o.Metadata)
	}
	return response, nil
}

// Client events of the Realtime API without a common command. Send them with
// RealtimeSession.SendRaw.

// RealtimeCreateItem adds an item, in Realtime API field names, to the
// conversation after previousItemID, or at the end when it is empty.
func RealtimeCreateItem(item map[string]any, previousItemID string) (json.RawMessage, error) {
	ev := map[string]any{"type": "conversation.item.create", "item": item}
	putNonEmpty(ev, "previous_item_id", previousItemID)
	return json.Marshal(ev)
}

// RealtimeDeleteItem removes an item from the conversation.
func RealtimeDeleteItem(itemID string) json.RawMessage {
	return mustRaw(map[string]any{"type": "conversation.item.delete", "item_id": itemID})
}

// RealtimeRetrieveItem asks for an item; it arrives as a raw
// conversation.item.retrieved event.
func RealtimeRetrieveItem(itemID string) json.RawMessage {
	return mustRaw(map[string]any{"type": "conversation.item.retrieve", "item_id": itemID})
}

// RealtimeTruncateItem cuts an assistant audio item at audioEnd, for example
// to keep only what the user heard.
func RealtimeTruncateItem(itemID string, contentIndex int, audioEnd time.Duration) json.RawMessage {
	return mustRaw(map[string]any{
		"type": "conversation.item.truncate", "item_id": itemID,
		"content_index": contentIndex, "audio_end_ms": audioEnd.Milliseconds(),
	})
}

// RealtimeClearInputAudio discards input audio not yet committed.
func RealtimeClearInputAudio() json.RawMessage {
	return mustRaw(map[string]any{"type": "input_audio_buffer.clear"})
}

// RealtimeMCPApproval answers an mcp_approval_request item from an MCP tool
// that requires approval; reason is optional.
func RealtimeMCPApproval(approvalRequestID string, approve bool, reason string) json.RawMessage {
	item := map[string]any{"type": "mcp_approval_response", "approval_request_id": approvalRequestID, "approve": approve}
	putNonEmpty(item, "reason", reason)
	return mustRaw(map[string]any{"type": "conversation.item.create", "item": item})
}

// mustRaw marshals the fixed events built here: strings, numbers, and maps of
// them always encode.
func mustRaw(v map[string]any) json.RawMessage {
	data, _ := json.Marshal(v)
	return data
}

func putNonEmpty(m map[string]any, key, value string) {
	if value != "" {
		m[key] = value
	}
}

func putMillis(m map[string]any, key string, d time.Duration) {
	if d > 0 {
		m[key] = d.Milliseconds()
	}
}
