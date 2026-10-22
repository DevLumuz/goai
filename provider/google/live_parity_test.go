package google

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/zendev-sh/goai/provider"
	"go.uber.org/goleak"
)

func ptr[T any](v T) *T { return &v }

// jsonEqual compares two values through their JSON encoding.
func jsonEqual(t *testing.T, got any, want string) {
	t.Helper()
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var g, w any
	_ = json.Unmarshal(data, &g)
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("bad expectation: %v", err)
	}
	if !reflect.DeepEqual(g, w) {
		t.Fatalf("got  %s\nwant %s", data, want)
	}
}

func TestLiveProviderOptions(t *testing.T) {
	cfg := LiveConfig{
		ResponseModalities:            []string{"TEXT"},
		Temperature:                   ptr(0.4),
		TopP:                          ptr(0.9),
		TopK:                          ptr(20.0),
		Seed:                          ptr(7),
		MediaResolution:               "MEDIA_RESOLUTION_LOW",
		LanguageCode:                  "es-US",
		SpeakerVoices:                 []LiveSpeakerVoice{{Speaker: "Ana", Voice: "Kore"}},
		ReplicatedVoice:               &LiveReplicatedVoice{MIMEType: "audio/wav", Sample: []byte{1}, ConsentAudio: []byte{2}, ConsentSignature: "sig"},
		Translation:                   &LiveTranslation{TargetLanguage: "fr", EchoTargetLanguage: ptr(false)},
		Thinking:                      &LiveThinking{Budget: ptr(512), IncludeThoughts: true, Level: "LOW"},
		AffectiveDialog:               ptr(true),
		ProactiveAudio:                ptr(true),
		SessionResumption:             &LiveSessionResumption{Handle: "h", Transparent: true},
		ContextCompression:            &LiveContextCompression{TriggerTokens: 25600, TargetTokens: 12800},
		InitialHistoryInClientContent: true,
		ExplicitVADSignal:             true,
		Avatar:                        &LiveAvatar{Name: "kai", ImageMIMEType: "image/png", Image: []byte{3}, AudioBitrate: 64000, VideoBitrate: 1000000},
		SafetySettings:                []LiveSafetySetting{{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_LOW_AND_ABOVE", Method: "SEVERITY"}},
		Labels:                        map[string]string{"team": "voice"},
		ActivityDetection:             &LiveActivityDetection{StartSensitivity: "START_SENSITIVITY_LOW", EndSensitivity: "END_SENSITIVITY_HIGH", PrefixPadding: 20 * time.Millisecond, SilenceDuration: 500 * time.Millisecond},
		TurnCoverage:                  "TURN_INCLUDES_ALL_INPUT",
		InterimTranscriptTimestamps:   ptr(true),
		InputTranscription: &LiveTranscription{
			LanguageCodes: []string{"es"}, LanguageAuto: true, LanguageHints: []string{"en"},
			CustomVocabulary: []string{"goai"}, AdaptationPhrases: []string{"VICLAW"},
			Diarization: ptr(true), WordTimestamps: ptr(true), Mode: "VERBATIM",
		},
		OutputTranscription: &LiveTranscription{},
	}
	jsonEqual(t, LiveProviderOptions(cfg), `{"google":{
		"generationConfig":{
			"responseModalities":["TEXT"],"temperature":0.4,"topP":0.9,"topK":20,"seed":7,
			"mediaResolution":"MEDIA_RESOLUTION_LOW","enableAffectiveDialog":true,
			"speechConfig":{"languageCode":"es-US",
				"multiSpeakerVoiceConfig":{"speakerVoiceConfigs":[{"speaker":"Ana","voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}}]},
				"voiceConfig":{"replicatedVoiceConfig":{"mimeType":"audio/wav","voiceSampleAudio":"AQ==","consentAudio":"Ag==","voiceConsentSignature":{"signature":"sig"}}}},
			"thinkingConfig":{"thinkingBudget":512,"includeThoughts":true,"thinkingLevel":"LOW"},
			"translationConfig":{"targetLanguageCode":"fr","echoTargetLanguage":false}},
		"proactivity":{"proactiveAudio":true},
		"sessionResumption":{"handle":"h","transparent":true},
		"contextWindowCompression":{"triggerTokens":"25600","slidingWindow":{"targetTokens":"12800"}},
		"historyConfig":{"initialHistoryInClientContent":true},
		"explicitVadSignal":true,
		"avatarConfig":{"avatarName":"kai","customizedAvatar":{"imageMimeType":"image/png","imageData":"Aw=="},"audioBitrateBps":64000,"videoBitrateBps":1000000},
		"safetySettings":[{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_LOW_AND_ABOVE","method":"SEVERITY"}],
		"labels":{"team":"voice"},
		"realtimeInputConfig":{
			"automaticActivityDetection":{"startOfSpeechSensitivity":"START_SENSITIVITY_LOW","endOfSpeechSensitivity":"END_SENSITIVITY_HIGH","prefixPaddingMs":20,"silenceDurationMs":500},
			"turnCoverage":"TURN_INCLUDES_ALL_INPUT","interimTranscriptTimestampEnabled":true},
		"inputAudioTranscription":{"languageCodes":["es"],"languageAuto":{},"languageHints":{"languageCodes":["en"]},
			"customVocabulary":["goai"],"adaptationPhrases":["VICLAW"],"diarization":true,"wordTimestamp":true,"mode":"VERBATIM"},
		"outputAudioTranscription":{}}}`)

	jsonEqual(t, LiveProviderOptions(LiveConfig{}), `{"google":{}}`)
	jsonEqual(t, LiveProviderOptions(LiveConfig{ContextCompression: &LiveContextCompression{}}),
		`{"google":{"contextWindowCompression":{"slidingWindow":{}}}}`)
}

func TestLive_LiveConfigMergesIntoSetup(t *testing.T) {
	cfg := LiveProviderOptions(LiveConfig{
		LanguageCode:       "es-US",
		Temperature:        ptr(0.2),
		ActivityDetection:  &LiveActivityDetection{SilenceDuration: time.Second},
		InputTranscription: &LiveTranscription{Diarization: ptr(true)},
	})
	setup, _, err := buildLiveSetup("models/m", provider.RealtimeParams{
		Voice:              "Kore",
		MaxOutputTokens:    100,
		TurnDetection:      provider.TurnDetection{Mode: provider.TurnModeClientVAD},
		InputTranscription: provider.TranscriptionConfig{Enabled: true, Language: "es"},
		ProviderOptions:    cfg,
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, setup, `{"model":"models/m",
		"generationConfig":{"responseModalities":["AUDIO"],"maxOutputTokens":100,"temperature":0.2,
			"speechConfig":{"languageCode":"es-US","voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}}},
		"realtimeInputConfig":{"automaticActivityDetection":{"disabled":true,"silenceDurationMs":1000}},
		"inputAudioTranscription":{"languageCodes":["es"],"diarization":true}}`)
}

func TestLive_VertexOnlySetupFields(t *testing.T) {
	for _, gopts := range []map[string]any{
		{"sessionResumption": map[string]any{"transparent": true}},
		{"explicitVadSignal": true},
	} {
		opts := map[string]any{"google": gopts}
		if _, err := mergeLiveProviderOptions(map[string]any{}, opts, false); !errors.Is(err, provider.ErrRealtimeUnsupported) {
			t.Errorf("Gemini API with %v: err = %v", gopts, err)
		}
		if _, err := mergeLiveProviderOptions(map[string]any{}, opts, true); err != nil {
			t.Errorf("Vertex AI with %v: err = %v", gopts, err)
		}
	}
}

// liveCommandSession opens a Gemini API session whose server forwards every
// client message.
func liveCommandSession(t *testing.T, params provider.RealtimeParams, serverMessages ...string) (provider.RealtimeConnection, <-chan map[string]json.RawMessage) {
	t.Helper()
	return liveCommandSessionOn(t, false, params, serverMessages...)
}

func liveCommandSessionOn(t *testing.T, vertex bool, params provider.RealtimeParams, serverMessages ...string) (provider.RealtimeConnection, <-chan map[string]json.RawMessage) {
	t.Helper()
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })
	received := make(chan map[string]json.RawMessage, 32)
	server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		acceptSetup(t, ctx, conn)
		for _, msg := range serverMessages {
			sendServerMessage(t, ctx, conn, msg)
		}
		for {
			_, data, err := conn.Read(ctx)
			if err != nil {
				return
			}
			var msg map[string]json.RawMessage
			_ = json.Unmarshal(data, &msg)
			received <- msg
		}
	})
	model := Live("m", WithAPIKey("k"), WithBaseURL(server.URL))
	if vertex {
		model = Live("m", WithTokenSource(provider.StaticToken("t")), WithVertex("p", "us-central1"), WithVertexBaseURL(server.URL))
	}
	conn, err := model.OpenRealtime(t.Context(), params)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	return conn, received
}

func nextClientMessage(t *testing.T, received <-chan map[string]json.RawMessage) map[string]json.RawMessage {
	t.Helper()
	select {
	case msg := <-received:
		return msg
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for a client message")
		return nil
	}
}

func TestLive_MediaAndAudioEndCommands(t *testing.T) {
	conn, received := liveCommandSession(t, provider.RealtimeParams{})
	ctx := t.Context()

	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeImageInput, Media: &provider.RealtimeMedia{MIMEType: "image/jpeg", Data: []byte{0xff, 0xd8}}}); err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, nextClientMessage(t, received), `{"realtimeInput":{"video":{"mimeType":"image/jpeg","data":"/9g="}}}`)

	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioEnd}); err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, nextClientMessage(t, received), `{"realtimeInput":{"audioStreamEnd":true}}`)

	for _, media := range []*provider.RealtimeMedia{nil, {MIMEType: "image/jpeg"}, {Data: []byte{1}}} {
		if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeImageInput, Media: media}); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
			t.Errorf("image %+v: err = %v", media, err)
		}
	}

	clientVAD, _ := liveCommandSession(t, provider.RealtimeParams{TurnDetection: provider.TurnDetection{Mode: provider.TurnModeClientVAD}})
	if err := clientVAD.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioEnd}); !errors.Is(err, provider.ErrRealtimeUnsupported) {
		t.Errorf("audioStreamEnd with client VAD: err = %v", err)
	}
}

func TestLive_ConversationAppend(t *testing.T) {
	conn, received := liveCommandSession(t, provider.RealtimeParams{})
	ctx := t.Context()

	conv := &provider.RealtimeConversation{RequestResponse: true, Messages: []provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "Hi"}}},
		{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartText, Text: "Hello"}}},
	}}
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeConversationAppend, Conversation: conv}); err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, nextClientMessage(t, received), `{"clientContent":{"turnComplete":true,"turns":[
		{"role":"user","parts":[{"text":"Hi"}]},{"role":"model","parts":[{"text":"Hello"}]}]}}`)

	conv.RequestResponse = false
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeConversationAppend, Conversation: conv}); err != nil {
		t.Fatal(err)
	}
	if msg := nextClientMessage(t, received); strings.Contains(string(msg["clientContent"]), "turnComplete") {
		t.Errorf("history without a response must not complete the turn: %s", msg["clientContent"])
	}

	for _, bad := range []*provider.RealtimeConversation{
		nil,
		{},
		{Messages: []provider.Message{{Role: provider.RoleSystem, Content: []provider.Part{{Type: provider.PartText, Text: "x"}}}}},
		{Messages: []provider.Message{{Role: provider.RoleUser}}},
	} {
		if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeConversationAppend, Conversation: bad}); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
			t.Errorf("conversation %+v: err = %v", bad, err)
		}
	}
}

func TestLive_SessionUpdate(t *testing.T) {
	gemini, _ := liveCommandSession(t, provider.RealtimeParams{})
	system := "x"
	err := gemini.Send(t.Context(), provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: &provider.RealtimeSessionUpdate{System: &system}})
	if !errors.Is(err, provider.ErrRealtimeUnsupported) {
		t.Fatalf("the Gemini API ignores contextUpdate, so it must fail: %v", err)
	}
	if provider.RealtimeCapabilitiesOf(Live("m")).RuntimeSessionUpdate || !provider.RealtimeCapabilitiesOf(Live("m", WithVertex("p", "l"))).RuntimeSessionUpdate {
		t.Fatal("RuntimeSessionUpdate must be Vertex AI only")
	}

	conn, received := liveCommandSessionOn(t, true, provider.RealtimeParams{},
		`{"toolCall":{"functionCalls":[{"id":"c1","name":"lookup","args":{}}]}}`)
	ctx := t.Context()
	nextEvent(t, conn) // ready
	nextEvent(t, conn) // response started
	nextEvent(t, conn) // tool call

	system = "Be brief."
	update := &provider.RealtimeSessionUpdate{
		System: &system,
		Tools: []provider.ToolDefinition{{
			Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`),
			ProviderDefinedOptions: LiveToolOptions(LiveNonBlocking, LiveSilent),
		}},
		ProviderOptions: map[string]any{"google": map[string]any{"futureField": 1}},
	}
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: update}); err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, nextClientMessage(t, received), `{"contextUpdate":{
		"systemInstruction":{"parts":[{"text":"Be brief."}]},
		"tools":{"tools":[{"functionDeclarations":[{"name":"lookup","description":"","behavior":"NON_BLOCKING","parameters":{"type":"object"}}]}]},
		"futureField":1}}`)

	// The new scheduling applies to results sent after the update.
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{ToolCallID: "c1", Output: json.RawMessage(`{}`)}}); err != nil {
		t.Fatal(err)
	}
	if msg := nextClientMessage(t, received); !strings.Contains(string(msg["toolResponse"]), `"scheduling":"SILENT"`) {
		t.Errorf("tool response = %s", msg["toolResponse"])
	}

	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: &provider.RealtimeSessionUpdate{
		System: ptr(""), Tools: []provider.ToolDefinition{},
	}}); err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, nextClientMessage(t, received), `{"contextUpdate":{"systemInstruction":{"parts":[]},"tools":{}}}`)

	for _, bad := range []*provider.RealtimeSessionUpdate{
		nil,
		{},
		{ProviderOptions: map[string]any{"google": map[string]any{"tools": []any{}}}},
		{Tools: []provider.ToolDefinition{{Name: "x", ProviderDefinedOptions: map[string]any{"behavior": "SOMETIMES"}}}},
	} {
		if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeSessionUpdateEvent, Update: bad}); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
			t.Errorf("update %+v: err = %v", bad, err)
		}
	}
}

func TestLive_ToolResultParity(t *testing.T) {
	conn, received := liveCommandSession(t, provider.RealtimeParams{},
		`{"toolCall":{"functionCalls":[{"id":"c1","name":"watch","args":{}}]}}`)
	ctx := t.Context()
	for range 3 {
		nextEvent(t, conn) // ready, response started, tool call
	}
	result := func(r provider.RealtimeToolResult) error {
		r.ToolCallID = "c1"
		return conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: &r})
	}

	if err := result(provider.RealtimeToolResult{
		Output:          json.RawMessage(`{"progress":50}`),
		Partial:         true,
		Media:           []provider.RealtimeMedia{{MIMEType: "image/png", Data: []byte{1}}},
		ProviderOptions: map[string]any{"google": map[string]any{"scheduling": "INTERRUPT"}},
	}); err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, nextClientMessage(t, received), `{"toolResponse":{"functionResponses":[{"id":"c1","name":"watch",
		"response":{"progress":50},"willContinue":true,"scheduling":"INTERRUPT",
		"parts":[{"inlineData":{"mimeType":"image/png","data":"AQ=="}}]}]}}`)

	// A partial result keeps the call open; the final one closes it.
	if err := result(provider.RealtimeToolResult{Output: json.RawMessage(`{"progress":100}`)}); err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, nextClientMessage(t, received), `{"toolResponse":{"functionResponses":[{"id":"c1","name":"watch","response":{"progress":100}}]}}`)
	if err := result(provider.RealtimeToolResult{Output: json.RawMessage(`{}`)}); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
		t.Errorf("result after the final one: err = %v", err)
	}

	open, _ := liveCommandSession(t, provider.RealtimeParams{}, `{"toolCall":{"functionCalls":[{"id":"c2","name":"x"}]}}`)
	for range 3 {
		nextEvent(t, open)
	}
	err := open.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{
		ToolCallID: "c2", ProviderOptions: map[string]any{"google": map[string]any{"id": "other"}},
	}})
	if !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
		t.Errorf("overriding the call id: err = %v", err)
	}
}

func TestLive_ServerMessageParity(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	partialCall := `{"toolCall":{"functionCalls":[{"id":"p1","name":"f","partialArgs":[{"jsonPath":"$.q","stringValue":"go"}],"willContinue":true}]}}`
	server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		_ = readClientMessage(t, ctx, conn)
		for _, msg := range []string{
			`{"setupComplete":{"sessionId":"s-1","voiceConsentSignature":{"signature":"sig-1"}}}`,
			`{"voiceActivity":{"voiceActivityType":"ACTIVITY_START","audioOffset":"0.200s"}}`,
			`{"voiceActivityDetectionSignal":{"vadSignalType":"VAD_SIGNAL_TYPE_SOS"}}`,
			`{"serverContent":{"interimInputTranscription":{"text":"hol"}}}`,
			`{"serverContent":{"inputTranscription":{"text":"hola","languageCode":"es","speakerLabel":"spk_1","startOffset":"0.2s","endOffset":"1s","words":[{"word":"hola"}]}}}`,
			`{"serverContent":{"inputTranscription":{"text":" amigo","endOffset":"1.5s","finished":true}}}`,
			`{"voiceActivityDetectionSignal":{"vadSignalType":"VAD_SIGNAL_TYPE_EOS"}}`,
			`{"voiceActivity":{"type":"ACTIVITY_END","audioOffset":"1.6s"}}`,
			`{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"image/png","data":"AQ=="}}]}}}`,
			`{"serverContent":{"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://go.dev","title":"Go"}}],"searchEntryPoint":{"renderedContent":"<div/>"}}}}`,
			`{"serverContent":{"urlContextMetadata":{"urlMetadata":[{"retrievedUrl":"https://a.dev","urlRetrievalStatus":"URL_RETRIEVAL_STATUS_SUCCESS"}]}}}`,
			partialCall,
			`{"serverContent":{"waitingForInput":true}}`,
			`{"serverContent":{"turnComplete":true,"turnCompleteReason":"PROHIBITED_INPUT_CONTENT","interactionStatus":"IN_PROGRESS"}}`,
			`{"serverContent":{"turnComplete":true,"turnCompleteReason":"MALFORMED_FUNCTION_CALL"}}`,
			`{"sessionResumptionUpdate":{"newHandle":"h2","resumable":true,"lastConsumedClientMessageIndex":"42"}}`,
			`{"usageMetadata":{"promptTokenCount":100,"toolUsePromptTokenCount":10,"cachedContentTokenCount":30,"candidatesTokenCount":9,"totalTokenCount":119,
				"promptTokensDetails":[{"modality":"AUDIO","tokenCount":100}],"candidatesTokensDetails":[{"modality":"AUDIO","tokenCount":9}],"trafficType":"ON_DEMAND","serviceTier":"standard"}}`,
		} {
			sendServerMessage(t, ctx, conn, msg)
		}
		drainUntilClose(ctx, conn)
	})

	conn, err := Live("m", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()

	google := func(ev provider.RealtimeEvent) map[string]any {
		m, _ := ev.ProviderMetadata["google"].(map[string]any)
		return m
	}
	expect := func(typ provider.RealtimeEventType) provider.RealtimeEvent {
		t.Helper()
		ev := nextEvent(t, conn)
		if ev.Type != typ {
			t.Fatalf("event = %s %+v, want %s", ev.Type, ev, typ)
		}
		return ev
	}

	if ev := expect(provider.RealtimeSessionReady); ev.SessionID != "s-1" || google(ev)["voiceConsentSignature"] != "sig-1" {
		t.Errorf("ready = %+v", ev)
	}
	if ev := expect(provider.RealtimeInputSpeechStarted); ev.AudioStart != 200*time.Millisecond {
		t.Errorf("speech started = %+v", ev)
	}
	expect(provider.RealtimeInputSpeechStarted) // VAD signal SOS
	if ev := expect(provider.RealtimeInputTranscript); !ev.IsInterim || ev.IsFinal || ev.Text != "hol" {
		t.Errorf("interim = %+v", ev)
	}
	ev := expect(provider.RealtimeInputTranscript)
	if ev.Text != "hola" || ev.Language != "es" || ev.Speaker != "spk_1" || ev.AudioStart != 200*time.Millisecond || ev.AudioEnd != time.Second || google(ev)["words"] == nil {
		t.Errorf("transcript delta = %+v", ev)
	}
	expect(provider.RealtimeInputTranscript) // " amigo" delta
	if ev := expect(provider.RealtimeInputTranscript); !ev.IsFinal || ev.Text != "hola amigo" || ev.Language != "es" || ev.Speaker != "spk_1" ||
		ev.AudioStart != 200*time.Millisecond || ev.AudioEnd != 1500*time.Millisecond {
		t.Errorf("finished transcript = %+v", ev)
	}
	expect(provider.RealtimeInputSpeechStopped) // VAD signal EOS
	if ev := expect(provider.RealtimeInputSpeechStopped); ev.AudioEnd != 1600*time.Millisecond {
		t.Errorf("speech stopped = %+v", ev)
	}
	expect(provider.RealtimeResponseStarted)
	if ev := expect(provider.RealtimeOutputMedia); ev.Media == nil || ev.Media.MIMEType != "image/png" || string(ev.Media.Data) != "\x01" {
		t.Errorf("output media = %+v", ev)
	}
	ev = expect(provider.RealtimeSources)
	if len(ev.Sources) != 1 || ev.Sources[0].URL != "https://go.dev" || ev.Sources[0].Title != "Go" {
		t.Errorf("grounding sources = %+v", ev.Sources)
	}
	if gm, _ := google(ev)["groundingMetadata"].(map[string]any); gm["searchEntryPoint"] == nil {
		t.Errorf("grounding metadata = %+v", ev.ProviderMetadata)
	}
	ev = expect(provider.RealtimeSources)
	if len(ev.Sources) != 1 || ev.Sources[0].URL != "https://a.dev" || google(provider.RealtimeEvent{ProviderMetadata: ev.Sources[0].ProviderMetadata})["urlRetrievalStatus"] != "URL_RETRIEVAL_STATUS_SUCCESS" {
		t.Errorf("url context = %+v", ev)
	}
	if ev := expect(provider.RealtimeProviderEvent); string(ev.Raw) != partialCall {
		t.Errorf("streamed function call must arrive raw: %s", ev.Raw)
	}
	expect(provider.RealtimeWaitingForInput)
	ev = expect(provider.RealtimeResponseFinished) // the image turn, blocked

	if ev.FinishReason != provider.FinishContentFilter || !ev.Pending || google(ev)["turnCompleteReason"] != "PROHIBITED_INPUT_CONTENT" || google(ev)["interactionStatus"] != "IN_PROGRESS" {
		t.Errorf("blocked turn = %+v", ev)
	}
	expect(provider.RealtimeResponseStarted) // a turn without output still ends in a pair
	if ev := expect(provider.RealtimeResponseFinished); ev.FinishReason != provider.FinishError || ev.Pending {
		t.Errorf("malformed call = %+v", ev)
	}
	if ev := expect(provider.RealtimeSessionResumeState); ev.ResumeToken != "h2" || google(ev)["lastConsumedClientMessageIndex"] != int64(42) {
		t.Errorf("resume state = %+v", ev)
	}
	ev = expect(provider.RealtimeUsageUpdated)
	if want := (provider.Usage{InputTokens: 80, CacheReadTokens: 30, OutputTokens: 9, TotalTokens: 119}); ev.Usage != want {
		t.Errorf("usage = %+v, want %+v", ev.Usage, want)
	}
	g := google(ev)
	if g["responseTokensDetails"] == nil || g["promptTokensDetails"] == nil || g["toolUsePromptTokenCount"] != 10 ||
		g["trafficType"] != "ON_DEMAND" || g["serviceTier"] != "standard" {
		t.Errorf("usage metadata = %+v", g)
	}
}

func TestLive_UnmappedServerSignals(t *testing.T) {
	c := &liveConn{calls: map[string]string{}}
	for _, raw := range []string{
		`{"voiceActivityDetectionSignal":{"vadSignalType":"FUTURE"}}`,
		`{"serverContent":{"interactionStatus":"IDLE"}}`,
		`{"toolCall":{"functionCalls":[{"id":"x","name":"f","willContinue":false}]}}`,
	} {
		var msg liveServerMessage
		if err := json.Unmarshal([]byte(raw), &msg); err != nil {
			t.Fatal(err)
		}
		if events, unmapped := c.translate(msg); !unmapped || len(events) != 0 {
			t.Errorf("%s: events %+v, unmapped %v", raw, events, unmapped)
		}
	}
	if len(c.calls) != 0 {
		t.Error("an undispatched call must not be registered")
	}
	if ev, ok := liveGroundingEvent(json.RawMessage(`[1]`)); ok {
		t.Errorf("invalid grounding = %+v", ev)
	}
	if ev, ok := liveURLContextEvent(nil); ok {
		t.Errorf("absent url context = %+v", ev)
	}
	if liveJSON(json.RawMessage(`{bad`)) != nil || liveDuration("bogus") != 0 {
		t.Error("invalid values must decode to zero")
	}
}

func TestLive_PendingTurnWithoutOutput(t *testing.T) {
	c := &liveConn{calls: map[string]string{}}
	var msg liveServerMessage
	_ = json.Unmarshal([]byte(`{"serverContent":{"turnComplete":true,"interactionStatus":"IN_PROGRESS"}}`), &msg)
	events, _ := c.translate(msg)
	if len(events) != 2 || events[0].Type != provider.RealtimeResponseStarted ||
		events[1].Type != provider.RealtimeResponseFinished || !events[1].Pending {
		t.Fatalf("events = %+v", events)
	}
	_ = json.Unmarshal([]byte(`{"serverContent":{"turnComplete":true,"interactionStatus":"IDLE"}}`), &msg)
	if events, _ := c.translate(msg); len(events) != 0 {
		t.Fatalf("an idle empty turn reports nothing: %+v", events)
	}
}
