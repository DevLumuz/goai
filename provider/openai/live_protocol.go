package openai

import "encoding/json"

// Wire types of the OpenAI GPT-Live API (/v1/live/sessions over WebSocket).
// Client events are built as maps; server events decode into one envelope.

type liveServerEvent struct {
	Type         string              `json:"type"`
	Session      *liveSessionInfo    `json:"session"`
	Delta        string              `json:"delta"`
	Delegation   *liveDelegationInfo `json:"delegation"`
	OffsetMs     int64               `json:"offset_ms"`
	DelegationID string              `json:"delegation_id"`
	Event        json.RawMessage     `json:"event"` // nested Responses event of response.event
	Usage        *liveUsage          `json:"usage"`
	Error        *liveWireError      `json:"error"`
	Reason       string              `json:"reason"`
	StartMs      int64               `json:"start_ms"`
	EndMs        int64               `json:"end_ms"`
	// ContextWindow is the share of the model's context in use.
	ContextWindow *struct {
		UsageRatio float64 `json:"usage_ratio"`
	} `json:"context_window"`
}

// liveSessionInfo is the session resource of session.started: the server's
// view of the delegation and audio, inherited by forks.
type liveSessionInfo struct {
	ID         string `json:"id"`
	Delegation *struct {
		Type string `json:"type"`
	} `json:"delegation"`
	Audio *struct {
		Format *struct {
			Type string `json:"type"`
			Rate int    `json:"rate"`
		} `json:"format"`
	} `json:"audio"`
}

type liveDelegationInfo struct {
	ID         string `json:"id"`
	Target     string `json:"target"`
	ResponseID string `json:"response_id"`
}

type liveUsage struct {
	Seconds float64 `json:"seconds"`
}

type liveWireError struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

// liveBackendEvent is the subset of a nested Responses event the adapter reads.
type liveBackendEvent struct {
	Type string `json:"type"`
	Item *struct {
		Type      string `json:"type"`
		CallID    string `json:"call_id"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"item"`
	Response *struct {
		ID    string         `json:"id"`
		Error *liveWireError `json:"error"`
	} `json:"response"`
	Error   *liveWireError `json:"error"`
	Code    string         `json:"code"`
	Message string         `json:"message"`
}
