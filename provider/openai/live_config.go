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

	goai "github.com/zendev-sh/goai"
)

// LiveConfig holds the GPT-Live session settings without a common realtime
// field, as in the official SDK's session config; zero values keep the
// defaults. Use it with goai.WithRealtimeProviderOptions(LiveProviderOptions(cfg)).
type LiveConfig struct {
	// Store keeps the session so it can be forked later.
	Store *bool
	// VoiceID selects a custom voice by ID instead of a named voice.
	VoiceID string
	// Input seeds the conversation, for example with a greeting to say first.
	Input []LiveInitialItem
	// Triggers run decisions on transcript events.
	Triggers []LiveDecisionTrigger
	// Decisions delegates to a Decisions backend instead of the default client.
	Decisions *LiveDecisions
}

// LiveInitialItem is one message of the initial conversation. Role is
// "developer", "user", or "assistant".
type LiveInitialItem struct {
	Role string
	Text string
}

// LiveDecisions configures Decisions delegation: a model answers Questions
// about the conversation.
type LiveDecisions struct {
	Model     string
	Questions []LiveDecisionQuestion
}

// LiveDecisionTrigger asks Questions whenever an event of EventType arrives,
// such as "session.input_transcript.delta".
type LiveDecisionTrigger struct {
	EventType string
	Questions []LiveDecisionQuestion
}

// LiveDecisionQuestion is a question for the Decisions model. Kind is
// "predicate" (yes or no), "choice" (one of Choices), or "score" (one of
// Levels).
type LiveDecisionQuestion struct {
	Kind         string
	Name         string
	Instructions string
	Choices      []LiveDecisionChoice
	Levels       []LiveDecisionLevel
}

// LiveDecisionChoice is a possible answer; Value is a string or a bool.
type LiveDecisionChoice struct {
	Value       any
	Description string
}

// LiveDecisionLevel is one level of a score question.
type LiveDecisionLevel struct {
	Label       string
	Description string
}

// LiveProviderOptions converts cfg to the ProviderOptions of a GPT-Live session.
func LiveProviderOptions(cfg LiveConfig) map[string]any {
	session := map[string]any{}
	if cfg.Store != nil {
		session["store"] = *cfg.Store
	}
	if cfg.VoiceID != "" {
		session["audio"] = map[string]any{"output": map[string]any{"voice": map[string]any{"id": cfg.VoiceID}}}
	}
	if len(cfg.Input) > 0 {
		items := make([]map[string]any, 0, len(cfg.Input))
		for _, it := range cfg.Input {
			partType := "input_text"
			if it.Role == "assistant" {
				partType = "output_text"
			}
			items = append(items, map[string]any{
				"type": "message", "role": it.Role,
				"content": []map[string]any{{"type": partType, "text": it.Text}},
			})
		}
		session["input"] = items
	}
	if len(cfg.Triggers) > 0 {
		triggers := make([]map[string]any, 0, len(cfg.Triggers))
		for _, t := range cfg.Triggers {
			triggers = append(triggers, map[string]any{
				"type":       "decisions",
				"event_type": t.EventType,
				"decisions":  map[string]any{"questions": liveQuestions(t.Questions)},
			})
		}
		session["triggers"] = triggers
	}
	if d := cfg.Decisions; d != nil {
		session["delegation"] = map[string]any{
			"type":      "decisions",
			"decisions": map[string]any{"model": d.Model, "questions": liveQuestions(d.Questions)},
		}
	}
	return session
}

func liveQuestions(questions []LiveDecisionQuestion) []map[string]any {
	out := make([]map[string]any, 0, len(questions))
	for _, q := range questions {
		question := map[string]any{"type": q.Kind, "instructions": q.Instructions}
		putNonEmpty(question, "name", q.Name)
		if len(q.Choices) > 0 {
			choices := make([]map[string]any, 0, len(q.Choices))
			for _, c := range q.Choices {
				choice := map[string]any{"value": c.Value}
				putNonEmpty(choice, "description", c.Description)
				choices = append(choices, choice)
			}
			question["choices"] = choices
		}
		if len(q.Levels) > 0 {
			levels := make([]map[string]any, 0, len(q.Levels))
			for _, l := range q.Levels {
				level := map[string]any{"label": l.Label}
				putNonEmpty(level, "description", l.Description)
				levels = append(levels, level)
			}
			question["levels"] = levels
		}
		out = append(out, question)
	}
	return out
}

// Client events of GPT-Live without a common command. Send them with
// RealtimeSession.SendRaw.

// LiveMuteInput stops the session from listening to the input audio.
func LiveMuteInput() json.RawMessage {
	return mustRaw(map[string]any{"type": "session.input_audio.mute"})
}

// LiveUnmuteInput resumes listening after LiveMuteInput.
func LiveUnmuteInput() json.RawMessage {
	return mustRaw(map[string]any{"type": "session.input_audio.unmute"})
}

// LivePlayFixture plays prerecorded audio, raw 24 kHz mono PCM16 without a
// container header, as the model's voice.
func LivePlayFixture(audio []byte) json.RawMessage {
	return mustRaw(map[string]any{"type": "session.output_audio.fixture.play", "audio": base64.StdEncoding.EncodeToString(audio)})
}

// LiveCreateDecision asks the Decisions backend to decide now.
func LiveCreateDecision() json.RawMessage {
	return mustRaw(map[string]any{"type": "decision.create"})
}

// LiveDeleteDecisionItem removes an item from the Decisions context.
func LiveDeleteDecisionItem(itemID string) json.RawMessage {
	return mustRaw(map[string]any{"type": "decision.item.delete", "item_id": itemID})
}

// LiveCreateResponseItem adds an item, in Responses API field names, to the
// Responses backend, for items without a common command.
func LiveCreateResponseItem(item map[string]any) (json.RawMessage, error) {
	return json.Marshal(map[string]any{"type": "response.item.create", "item": item})
}

// LiveSessionContent downloads the recorded content of a stored GPT-Live
// session. The caller closes the body; contentType is the server's media type.
func LiveSessionContent(ctx context.Context, sessionID string, opts ...Option) (body io.ReadCloser, contentType string, err error) {
	if sessionID == "" {
		return nil, "", errors.New("openai: live session ID is required")
	}
	o := resolveConnectOptions(opts)
	if o.tokenSource == nil {
		return nil, "", errors.New("goai: no API key or token source configured")
	}
	token, err := o.tokenSource.Token(ctx)
	if err != nil {
		return nil, "", fmt.Errorf("resolving auth token: %w", err)
	}
	endpoint := strings.TrimRight(o.baseURL, "/") + "/live/sessions/" + url.PathEscape(sessionID) + "/content"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, "", fmt.Errorf("openai: building content request: %w", err)
	}
	for k, v := range o.headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", "application/binary")
	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("openai: downloading live session content: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, liveMaxErrorBody))
		_ = resp.Body.Close()
		return nil, "", goai.ParseHTTPErrorWithHeaders("openai", resp.StatusCode, data, resp.Header)
	}
	return resp.Body, resp.Header.Get("Content-Type"), nil
}
