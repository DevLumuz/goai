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
}

type liveSessionInfo struct {
	ID string `json:"id"`
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
