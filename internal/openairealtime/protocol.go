package openairealtime

import "encoding/json"

// Wire types of the OpenAI Realtime API (GA interface). Client events are
// built as maps; server events decode into one flat envelope because every
// event type uses a small, distinct subset of these fields.

type serverEvent struct {
	Type         string          `json:"type"`
	Session      *wireSession    `json:"session"`
	Error        *wireError      `json:"error"`
	ItemID       string          `json:"item_id"`
	ResponseID   string          `json:"response_id"`
	Delta        string          `json:"delta"`
	Transcript   string          `json:"transcript"`
	CallID       string          `json:"call_id"`
	Name         string          `json:"name"`
	Arguments    string          `json:"arguments"`
	Response     *wireResponse   `json:"response"`
	RateLimits   []wireRateLimit `json:"rate_limits"`
	AudioStartMs int64           `json:"audio_start_ms"`
	AudioEndMs   int64           `json:"audio_end_ms"`
	Text         string          `json:"text"`
	Logprobs     json.RawMessage `json:"logprobs"`
	Usage        json.RawMessage `json:"usage"` // transcription usage
	// Diarized transcription segments carry their speaker and time span.
	ID      string  `json:"id"`
	Speaker string  `json:"speaker"`
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
}

type wireSession struct {
	ID string `json:"id"`
}

type wireError struct {
	EventID string `json:"event_id"`
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param"`
}

type wireResponse struct {
	ID string `json:"id"`
	// ConversationID is null for a response outside the default conversation.
	ConversationID json.RawMessage    `json:"conversation_id"`
	Metadata       map[string]any     `json:"metadata"`
	Status         string             `json:"status"`
	StatusDetails  *wireStatusDetails `json:"status_details"`
	Output         []wireOutputItem   `json:"output"`
	Usage          *wireUsage         `json:"usage"`
}

type wireStatusDetails struct {
	Type   string     `json:"type"`
	Reason string     `json:"reason"`
	Error  *wireError `json:"error"`
}

type wireOutputItem struct {
	Type string `json:"type"`
}

type wireUsage struct {
	TotalTokens        int             `json:"total_tokens"`
	InputTokens        int             `json:"input_tokens"`
	OutputTokens       int             `json:"output_tokens"`
	InputTokenDetails  json.RawMessage `json:"input_token_details"`
	OutputTokenDetails json.RawMessage `json:"output_token_details"`
}

func (u *wireUsage) cachedTokens() int {
	var details struct {
		CachedTokens int `json:"cached_tokens"`
	}
	_ = json.Unmarshal(u.InputTokenDetails, &details)
	return details.CachedTokens
}

type wireRateLimit struct {
	Name         string  `json:"name"`
	Limit        int     `json:"limit"`
	Remaining    int     `json:"remaining"`
	ResetSeconds float64 `json:"reset_seconds"`
}

// clientEvent is one outgoing JSON event.
type clientEvent map[string]any

func marshalEvent(ev clientEvent) ([]byte, error) { return json.Marshal(ev) }
