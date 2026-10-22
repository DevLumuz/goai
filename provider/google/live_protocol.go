package google

import "encoding/json"

// Wire types of the Gemini Live API (BidiGenerateContent over WebSocket). They
// stay private: callers only see provider.RealtimeModel and its events.

// liveClientMessage carries exactly one of its fields per WebSocket message.
type liveClientMessage struct {
	Setup         map[string]any     `json:"setup,omitempty"`
	ClientContent *liveClientContent `json:"clientContent,omitempty"`
	RealtimeInput *liveRealtimeInput `json:"realtimeInput,omitempty"`
	ToolResponse  *liveToolResponse  `json:"toolResponse,omitempty"`
	ContextUpdate map[string]any     `json:"contextUpdate,omitempty"`
}

type liveClientContent struct {
	Turns        []map[string]any `json:"turns"`
	TurnComplete bool             `json:"turnComplete,omitempty"`
}

// liveBlob is inline media; encoding/json writes Data as standard base64.
type liveBlob struct {
	MimeType string `json:"mimeType"`
	Data     []byte `json:"data"`
}

type liveRealtimeInput struct {
	Audio          *liveBlob `json:"audio,omitempty"`
	Video          *liveBlob `json:"video,omitempty"`
	Text           string    `json:"text,omitempty"`
	AudioStreamEnd bool      `json:"audioStreamEnd,omitempty"`
	ActivityStart  *struct{} `json:"activityStart,omitempty"`
	ActivityEnd    *struct{} `json:"activityEnd,omitempty"`
}

// liveToolResponse holds functionResponse objects built as maps, so provider
// options can merge into them.
type liveToolResponse struct {
	FunctionResponses []map[string]any `json:"functionResponses"`
}

// liveServerMessage holds usageMetadata/voiceActivity plus one message type.
type liveServerMessage struct {
	SetupComplete           *liveSetupComplete           `json:"setupComplete"`
	ServerContent           *liveServerContent           `json:"serverContent"`
	ToolCall                *liveToolCall                `json:"toolCall"`
	ToolCallCancellation    *liveToolCallCancellation    `json:"toolCallCancellation"`
	GoAway                  *liveGoAway                  `json:"goAway"`
	SessionResumptionUpdate *liveSessionResumptionUpdate `json:"sessionResumptionUpdate"`
	UsageMetadata           *liveUsageMetadata           `json:"usageMetadata"`
	VoiceActivity           *liveVoiceActivity           `json:"voiceActivity"`
	VADSignal               *liveVADSignal               `json:"voiceActivityDetectionSignal"`
	// Error is an API error sent as a message instead of a close frame.
	Error *liveError `json:"error"`
}

type liveSetupComplete struct {
	SessionID             string `json:"sessionId"`
	VoiceConsentSignature *struct {
		Signature string `json:"signature"`
	} `json:"voiceConsentSignature"`
}

type liveError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

type liveServerContent struct {
	ModelTurn                 *liveContent       `json:"modelTurn"`
	TurnComplete              bool               `json:"turnComplete"`
	GenerationComplete        bool               `json:"generationComplete"`
	Interrupted               bool               `json:"interrupted"`
	InputTranscription        *liveTranscription `json:"inputTranscription"`
	InterimInputTranscription *liveTranscription `json:"interimInputTranscription"`
	OutputTranscription       *liveTranscription `json:"outputTranscription"`
	GroundingMetadata         json.RawMessage    `json:"groundingMetadata"`
	URLContextMetadata        json.RawMessage    `json:"urlContextMetadata"`
	TurnCompleteReason        string             `json:"turnCompleteReason"`
	WaitingForInput           bool               `json:"waitingForInput"`
	InteractionStatus         string             `json:"interactionStatus"`
}

type liveContent struct {
	Role  string     `json:"role"`
	Parts []livePart `json:"parts"`
}

type livePart struct {
	Text       string    `json:"text"`
	Thought    bool      `json:"thought"`
	InlineData *liveBlob `json:"inlineData"`
}

// liveTranscription offsets are protobuf durations such as "1.5s".
type liveTranscription struct {
	Text         string          `json:"text"`
	Finished     bool            `json:"finished"`
	LanguageCode string          `json:"languageCode"`
	SpeakerLabel string          `json:"speakerLabel"`
	Words        json.RawMessage `json:"words"`
	StartOffset  string          `json:"startOffset"`
	EndOffset    string          `json:"endOffset"`
}

type liveToolCall struct {
	FunctionCalls []liveFunctionCall `json:"functionCalls"`
}

// liveFunctionCall carries partialArgs/willContinue when arguments stream;
// Live does not document that mode, so such calls are not dispatched.
type liveFunctionCall struct {
	ID           string          `json:"id"`
	Name         string          `json:"name"`
	Args         json.RawMessage `json:"args"`
	PartialArgs  json.RawMessage `json:"partialArgs"`
	WillContinue *bool           `json:"willContinue"`
}

type liveToolCallCancellation struct {
	IDs []string `json:"ids"`
}

type liveGoAway struct {
	TimeLeft string `json:"timeLeft"`
}

type liveSessionResumptionUpdate struct {
	NewHandle string `json:"newHandle"`
	Resumable bool   `json:"resumable"`
	// LastConsumedClientMessageIndex is an int64, which protobuf JSON writes
	// as a string.
	LastConsumedClientMessageIndex json.Number `json:"lastConsumedClientMessageIndex"`
}

// liveUsageMetadata uses the Gemini API names; Vertex AI writes candidates*
// for response*.
type liveUsageMetadata struct {
	PromptTokenCount           int             `json:"promptTokenCount"`
	CachedContentTokenCount    int             `json:"cachedContentTokenCount"`
	ResponseTokenCount         int             `json:"responseTokenCount"`
	CandidatesTokenCount       int             `json:"candidatesTokenCount"`
	ToolUsePromptTokenCount    int             `json:"toolUsePromptTokenCount"`
	ThoughtsTokenCount         int             `json:"thoughtsTokenCount"`
	TotalTokenCount            int             `json:"totalTokenCount"`
	PromptTokensDetails        json.RawMessage `json:"promptTokensDetails"`
	CacheTokensDetails         json.RawMessage `json:"cacheTokensDetails"`
	ResponseTokensDetails      json.RawMessage `json:"responseTokensDetails"`
	CandidatesTokensDetails    json.RawMessage `json:"candidatesTokensDetails"`
	ToolUsePromptTokensDetails json.RawMessage `json:"toolUsePromptTokensDetails"`
	TrafficType                string          `json:"trafficType"`
	ServiceTier                string          `json:"serviceTier"`
}

// liveVoiceActivity names its type "type" on the wire; voiceActivityType is
// the official SDK's field name, accepted as well.
type liveVoiceActivity struct {
	Type              string `json:"type"`
	VoiceActivityType string `json:"voiceActivityType"`
	AudioOffset       string `json:"audioOffset"`
}

type liveVADSignal struct {
	Type string `json:"vadSignalType"`
}
