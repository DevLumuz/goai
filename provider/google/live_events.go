package google

import (
	"cmp"
	"encoding/json"
	"strings"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// liveTranscript accumulates one side's transcript for the current turn.
type liveTranscript struct {
	text       strings.Builder
	language   string
	speaker    string
	start, end time.Duration
}

// add records a transcription message and returns its events: the delta, and
// the full transcript when the server marks it finished.
func (t *liveTranscript) add(typ provider.RealtimeEventType, tr *liveTranscription) []provider.RealtimeEvent {
	var events []provider.RealtimeEvent
	if tr.Text != "" {
		ev := liveTranscriptEvent(typ, tr)
		if t.text.Len() == 0 {
			t.start = ev.AudioStart
		}
		t.text.WriteString(tr.Text)
		t.language = cmp.Or(ev.Language, t.language)
		t.speaker = cmp.Or(ev.Speaker, t.speaker)
		t.end = max(ev.AudioEnd, t.end)
		events = append(events, ev)
	}
	if tr.Finished {
		events = append(events, t.finish(typ)...)
	}
	return events
}

// finish returns the final transcript, if any, and resets the buffer.
func (t *liveTranscript) finish(typ provider.RealtimeEventType) []provider.RealtimeEvent {
	defer t.reset()
	if t.text.Len() == 0 {
		return nil
	}
	return []provider.RealtimeEvent{{
		Type: typ, Text: t.text.String(), IsFinal: true,
		Language: t.language, Speaker: t.speaker, AudioStart: t.start, AudioEnd: t.end,
	}}
}

func (t *liveTranscript) reset() {
	t.text.Reset()
	t.language, t.speaker, t.start, t.end = "", "", 0, 0
}

func liveTranscriptEvent(typ provider.RealtimeEventType, tr *liveTranscription) provider.RealtimeEvent {
	ev := provider.RealtimeEvent{
		Type:       typ,
		Text:       tr.Text,
		Language:   tr.LanguageCode,
		Speaker:    tr.SpeakerLabel,
		AudioStart: liveDuration(tr.StartOffset),
		AudioEnd:   liveDuration(tr.EndOffset),
	}
	if words := liveJSON(tr.Words); words != nil {
		ev.ProviderMetadata = liveMetadata("words", words)
	}
	return ev
}

// liveDuration parses a protobuf duration such as "1.5s"; invalid values are zero.
func liveDuration(s string) time.Duration {
	d, _ := time.ParseDuration(s)
	return d
}

// liveJSON decodes a raw field for ProviderMetadata; absent or invalid is nil.
func liveJSON(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return nil
	}
	return v
}

func liveMetadata(kv ...any) map[string]any {
	google := make(map[string]any, len(kv)/2)
	for i := 0; i+1 < len(kv); i += 2 {
		google[kv[i].(string)] = kv[i+1]
	}
	return map[string]any{"google": google}
}

// liveContentFilterReasons are turnCompleteReason values that block the turn
// for safety or policy.
var liveContentFilterReasons = map[string]bool{
	"PROHIBITED_INPUT_CONTENT":                       true,
	"IMAGE_PROHIBITED_INPUT_CONTENT":                 true,
	"INPUT_TEXT_CONTAIN_PROMINENT_PERSON_PROHIBITED": true,
	"INPUT_IMAGE_CELEBRITY":                          true,
	"INPUT_IMAGE_PHOTO_REALISTIC_CHILD_PROHIBITED":   true,
	"INPUT_TEXT_NCII_PROHIBITED":                     true,
	"INPUT_OTHER":                                    true,
	"INPUT_IP_PROHIBITED":                            true,
	"BLOCKLIST":                                      true,
}

// translate maps one server message to common events and reports whether part
// of it has no mapping. It runs only on the goroutine that owns the turn state:
// the opener during setup, then the reader.
func (c *liveConn) translate(msg liveServerMessage) (events []provider.RealtimeEvent, unmapped bool) {
	startResponse := func() {
		if !c.responding {
			c.responding = true
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeResponseStarted})
		}
	}

	if msg.Error != nil {
		events = append(events, provider.RealtimeEvent{Type: provider.RealtimeError, Error: msg.Error.err()})
	}

	if va := msg.VoiceActivity; va != nil {
		offset := liveDuration(va.AudioOffset)
		switch cmp.Or(va.Type, va.VoiceActivityType) {
		case "ACTIVITY_START":
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeInputSpeechStarted, AudioStart: offset})
		case "ACTIVITY_END":
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeInputSpeechStopped, AudioEnd: offset})
		default:
			unmapped = true
		}
	}
	if sig := msg.VADSignal; sig != nil {
		switch sig.Type {
		case "VAD_SIGNAL_TYPE_SOS":
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeInputSpeechStarted})
		case "VAD_SIGNAL_TYPE_EOS":
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeInputSpeechStopped})
		default:
			unmapped = true
		}
	}

	if sc := msg.ServerContent; sc != nil {
		if tr := sc.InterimInputTranscription; tr != nil && tr.Text != "" {
			ev := liveTranscriptEvent(provider.RealtimeInputTranscript, tr)
			ev.IsInterim = true
			events = append(events, ev)
		}
		if tr := sc.InputTranscription; tr != nil {
			events = append(events, c.inputTranscript.add(provider.RealtimeInputTranscript, tr)...)
		}
		if turn := sc.ModelTurn; turn != nil {
			for _, part := range turn.Parts {
				switch {
				case part.Thought:
					if part.Text != "" {
						startResponse()
						events = append(events, provider.RealtimeEvent{Type: provider.RealtimeReasoningDelta, Text: part.Text})
					}
				case part.InlineData != nil && strings.HasPrefix(part.InlineData.MimeType, "audio/"):
					startResponse()
					c.turnHadOutput = true
					events = append(events, provider.RealtimeEvent{
						Type:        provider.RealtimeOutputAudio,
						Audio:       part.InlineData.Data,
						AudioFormat: liveOutputFormat,
					})
				case part.InlineData != nil && (strings.HasPrefix(part.InlineData.MimeType, "image/") ||
					strings.HasPrefix(part.InlineData.MimeType, "video/")):
					startResponse()
					c.turnHadOutput = true
					events = append(events, provider.RealtimeEvent{
						Type:  provider.RealtimeOutputMedia,
						Media: &provider.RealtimeMedia{MIMEType: part.InlineData.MimeType, Data: part.InlineData.Data},
					})
				case part.Text != "":
					startResponse()
					c.turnHadOutput = true
					events = append(events, provider.RealtimeEvent{Type: provider.RealtimeTextDelta, Text: part.Text})
				default:
					startResponse()
					c.turnHadOutput = true
					unmapped = true
				}
			}
		}
		if tr := sc.OutputTranscription; tr != nil {
			if tr.Text != "" {
				startResponse()
			}
			events = append(events, c.outputTranscript.add(provider.RealtimeOutputTranscript, tr)...)
		}
		if ev, ok := liveGroundingEvent(sc.GroundingMetadata); ok {
			events = append(events, ev)
		}
		if ev, ok := liveURLContextEvent(sc.URLContextMetadata); ok {
			events = append(events, ev)
		}
		if sc.Interrupted {
			c.interrupted = true
		}
		if sc.WaitingForInput {
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeWaitingForInput})
		}
		if sc.GenerationComplete {
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeGenerationComplete})
		}
		if sc.TurnComplete {
			events = append(events, c.finishTurn(sc.TurnCompleteReason, sc.InteractionStatus)...)
		} else if sc.TurnCompleteReason != "" || sc.InteractionStatus != "" {
			unmapped = true // documented only alongside turnComplete
		}
	}

	if tc := msg.ToolCall; tc != nil {
		for _, call := range tc.FunctionCalls {
			if len(call.PartialArgs) > 0 || call.WillContinue != nil {
				unmapped = true
				continue
			}
			startResponse()
			c.turnHadTools = true
			c.callsMu.Lock()
			c.calls[call.ID] = call.Name
			c.callsMu.Unlock()
			input := call.Args
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			events = append(events, provider.RealtimeEvent{
				Type:       provider.RealtimeToolCallEvent,
				ToolCallID: call.ID,
				ToolCall:   &provider.RealtimeToolCall{ID: call.ID, Name: call.Name, Input: input},
			})
		}
	}

	if cancel := msg.ToolCallCancellation; cancel != nil {
		for _, id := range cancel.IDs {
			c.callsMu.Lock()
			delete(c.calls, id)
			c.callsMu.Unlock()
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeToolCallCancelled, ToolCallID: id})
		}
	}

	if ga := msg.GoAway; ga != nil {
		events = append(events, provider.RealtimeEvent{Type: provider.RealtimeSessionGoAway, TimeToClose: liveDuration(ga.TimeLeft)})
	}

	if ru := msg.SessionResumptionUpdate; ru != nil {
		// An empty token means the session cannot be resumed at this point.
		ev := provider.RealtimeEvent{Type: provider.RealtimeSessionResumeState}
		if ru.Resumable {
			ev.ResumeToken = ru.NewHandle
		}
		if index, err := ru.LastConsumedClientMessageIndex.Int64(); err == nil {
			ev.ProviderMetadata = liveMetadata("lastConsumedClientMessageIndex", index)
		}
		events = append(events, ev)
	}

	if um := msg.UsageMetadata; um != nil {
		events = append(events, liveUsageEvent(um))
	}
	return events, unmapped
}

// liveUsageEvent maps usage like the REST provider. Tool results are prompt
// input billed apart from promptTokenCount, so they count as input tokens.
func liveUsageEvent(um *liveUsageMetadata) provider.RealtimeEvent {
	ev := provider.RealtimeEvent{Type: provider.RealtimeUsageUpdated, Usage: provider.Usage{
		CacheReadTokens: um.CachedContentTokenCount,
		InputTokens:     max(um.PromptTokenCount+um.ToolUsePromptTokenCount-um.CachedContentTokenCount, 0),
		OutputTokens:    cmp.Or(um.ResponseTokenCount, um.CandidatesTokenCount),
		ReasoningTokens: um.ThoughtsTokenCount,
		TotalTokens:     um.TotalTokenCount,
	}}
	google := map[string]any{}
	responseDetails := um.ResponseTokensDetails
	if len(responseDetails) == 0 {
		responseDetails = um.CandidatesTokensDetails
	}
	for key, raw := range map[string]json.RawMessage{
		"promptTokensDetails":        um.PromptTokensDetails,
		"cacheTokensDetails":         um.CacheTokensDetails,
		"responseTokensDetails":      responseDetails,
		"toolUsePromptTokensDetails": um.ToolUsePromptTokensDetails,
	} {
		if v := liveJSON(raw); v != nil {
			google[key] = v
		}
	}
	if um.ToolUsePromptTokenCount > 0 {
		google["toolUsePromptTokenCount"] = um.ToolUsePromptTokenCount
	}
	if um.TrafficType != "" {
		google["trafficType"] = um.TrafficType
	}
	if um.ServiceTier != "" {
		google["serviceTier"] = um.ServiceTier
	}
	if len(google) > 0 {
		ev.ProviderMetadata = map[string]any{"google": google}
	}
	return ev
}

// liveGroundingEvent reports grounding sources; the full metadata, including
// the search suggestions Google requires apps to show, stays in ProviderMetadata.
func liveGroundingEvent(raw json.RawMessage) (provider.RealtimeEvent, bool) {
	var gm groundingMetadata
	if len(raw) == 0 || json.Unmarshal(raw, &gm) != nil {
		return provider.RealtimeEvent{}, false
	}
	return provider.RealtimeEvent{
		Type:             provider.RealtimeSources,
		Sources:          extractGroundingSources(&gm),
		ProviderMetadata: liveMetadata("groundingMetadata", liveJSON(raw)),
	}, true
}

// liveURLContextEvent reports the URLs the url context tool retrieved, each
// with its retrieval status.
func liveURLContextEvent(raw json.RawMessage) (provider.RealtimeEvent, bool) {
	var meta struct {
		URLMetadata []struct {
			RetrievedURL       string `json:"retrievedUrl"`
			URLRetrievalStatus string `json:"urlRetrievalStatus"`
		} `json:"urlMetadata"`
	}
	if len(raw) == 0 || json.Unmarshal(raw, &meta) != nil {
		return provider.RealtimeEvent{}, false
	}
	sources := make([]provider.Source, 0, len(meta.URLMetadata))
	for _, u := range meta.URLMetadata {
		sources = append(sources, provider.Source{
			Type:             "url",
			URL:              u.RetrievedURL,
			ProviderMetadata: liveMetadata("urlRetrievalStatus", u.URLRetrievalStatus),
		})
	}
	return provider.RealtimeEvent{
		Type:             provider.RealtimeSources,
		Sources:          sources,
		ProviderMetadata: liveMetadata("urlContextMetadata", liveJSON(raw)),
	}, true
}

// finishTurn closes the current model turn: final transcripts not yet marked
// finished, then the response finish with the reason the turn ended.
func (c *liveConn) finishTurn(reason, status string) []provider.RealtimeEvent {
	events := c.inputTranscript.finish(provider.RealtimeInputTranscript)
	events = append(events, c.outputTranscript.finish(provider.RealtimeOutputTranscript)...)
	// The interaction continues while a function runs or awaits a result.
	pending := status == "IN_PROGRESS" || status == "REQUIRES_ACTION"
	if !c.responding && (c.interrupted || reason != "" || pending) {
		// A turn without output still ends, interrupted, blocked, or pending:
		// keep the pair.
		c.responding = true
		events = append(events, provider.RealtimeEvent{Type: provider.RealtimeResponseStarted})
	}
	if c.responding {
		finish := provider.FinishStop
		switch {
		case c.interrupted:
			finish = provider.FinishInterrupted
		case liveContentFilterReasons[reason]:
			finish = provider.FinishContentFilter
		case reason == "MALFORMED_FUNCTION_CALL":
			finish = provider.FinishError
		case c.turnHadTools && !c.turnHadOutput:
			finish = provider.FinishToolCalls
		}
		ev := provider.RealtimeEvent{
			Type:         provider.RealtimeResponseFinished,
			FinishReason: finish,
			Pending:      pending,
		}
		google := map[string]any{}
		if reason != "" {
			google["turnCompleteReason"] = reason
		}
		if status != "" {
			google["interactionStatus"] = status
		}
		if len(google) > 0 {
			ev.ProviderMetadata = map[string]any{"google": google}
		}
		events = append(events, ev)
	}
	c.responding, c.interrupted, c.turnHadOutput, c.turnHadTools = false, false, false, false
	return events
}
