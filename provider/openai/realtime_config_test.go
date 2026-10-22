package openai

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/zendev-sh/goai/internal/openairealtime"
	"github.com/zendev-sh/goai/provider"
)

func ptrTo[T any](v T) *T { return &v }

func assertJSON(t *testing.T, got any, want string) {
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

func TestRealtimeProviderOptions(t *testing.T) {
	assertJSON(t, RealtimeProviderOptions(RealtimeConfig{
		OutputModalities: []string{"text"},
		VoiceID:          "voice_123",
		Speed:            ptrTo(1.1),
		NoiseReduction:   "far_field",
		TurnDetection: &RealtimeTurnDetection{
			Threshold: ptrTo(0.6), PrefixPadding: 300 * time.Millisecond,
			SilenceDuration: 500 * time.Millisecond, IdleTimeout: 6 * time.Second,
		},
		Transcription:     &RealtimeTranscription{Model: "gpt-4o-transcribe", Keywords: []string{"goai"}, Languages: []string{"es"}, Delay: "low"},
		Include:           []string{"item.input_audio_transcription.logprobs"},
		Prompt:            &RealtimePrompt{ID: "pmpt_1", Version: "2", Variables: map[string]any{"name": "Ana"}},
		Tracing:           &RealtimeTracing{WorkflowName: "support", GroupID: "g1", Metadata: map[string]any{"k": "v"}},
		Truncation:        &RealtimeTruncation{RetentionRatio: 0.8, PostInstructionsTokens: 4000},
		ReasoningEffort:   "low",
		ParallelToolCalls: ptrTo(false),
	}), `{
		"output_modalities":["text"],
		"audio":{
			"input":{"noise_reduction":{"type":"far_field"},
				"turn_detection":{"type":"server_vad","threshold":0.6,"prefix_padding_ms":300,"silence_duration_ms":500,"idle_timeout_ms":6000},
				"transcription":{"model":"gpt-4o-transcribe","keywords":["goai"],"languages":["es"],"delay":"low"}},
			"output":{"voice":{"id":"voice_123"},"speed":1.1}},
		"include":["item.input_audio_transcription.logprobs"],
		"prompt":{"id":"pmpt_1","version":"2","variables":{"name":"Ana"}},
		"tracing":{"workflow_name":"support","group_id":"g1","metadata":{"k":"v"}},
		"truncation":{"type":"retention_ratio","retention_ratio":0.8,"token_limits":{"post_instructions":4000}},
		"reasoning":{"effort":"low"},
		"parallel_tool_calls":false}`)

	assertJSON(t, RealtimeProviderOptions(RealtimeConfig{}), `{}`)
	assertJSON(t, RealtimeProviderOptions(RealtimeConfig{
		TurnDetection: &RealtimeTurnDetection{Type: "semantic_vad", Eagerness: "high"},
		Tracing:       &RealtimeTracing{},
		Truncation:    &RealtimeTruncation{Disabled: true},
	}), `{"audio":{"input":{"turn_detection":{"type":"semantic_vad","eagerness":"high"}}},"tracing":"auto","truncation":"disabled"}`)
	assertJSON(t, RealtimeProviderOptions(RealtimeConfig{Truncation: &RealtimeTruncation{}}), `{"truncation":"auto"}`)
}

func TestRealtimeResponse(t *testing.T) {
	got, err := RealtimeResponse(RealtimeResponseOptions{
		OutOfBand:        true,
		InputItemIDs:     []string{"item_1"},
		Input:            []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "Classify."}}}},
		Instructions:     "Answer with one label.",
		Tools:            []provider.ToolDefinition{{Name: "tag", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice:       "required",
		OutputModalities: []string{"text"},
		MaxOutputTokens:  20,
		Voice:            "marin",
		ReasoningEffort:  "minimal",
		Prompt:           &RealtimePrompt{ID: "pmpt_2"},
		Metadata:         map[string]string{"purpose": "classify"},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertJSON(t, got, `{
		"conversation":"none",
		"input":[{"type":"item_reference","id":"item_1"},{"type":"message","role":"user","content":[{"type":"input_text","text":"Classify."}]}],
		"instructions":"Answer with one label.",
		"tools":[{"type":"function","name":"tag","description":"","parameters":{"type":"object"}}],
		"tool_choice":"required","output_modalities":["text"],"max_output_tokens":20,
		"audio":{"output":{"voice":"marin"}},"reasoning":{"effort":"minimal"},
		"prompt":{"id":"pmpt_2"},"metadata":{"purpose":"classify"}}`)

	if got, err := RealtimeResponse(RealtimeResponseOptions{Tools: []provider.ToolDefinition{}}); err != nil {
		t.Fatal(err)
	} else {
		assertJSON(t, got, `{"tools":[]}`)
	}
	if _, err := RealtimeResponse(RealtimeResponseOptions{Input: []provider.Message{{Role: provider.RoleUser}}}); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
		t.Errorf("empty input: %v", err)
	}
	if _, err := RealtimeResponse(RealtimeResponseOptions{Tools: []provider.ToolDefinition{{Name: "x", ProviderDefinedType: "web_search"}}}); !errors.Is(err, provider.ErrRealtimeUnsupported) {
		t.Errorf("unsupported tool: %v", err)
	}
}

func TestRealtimeRawCommands(t *testing.T) {
	item, err := RealtimeCreateItem(map[string]any{"type": "message", "role": "user"}, "item_0")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		got  json.RawMessage
		want string
	}{
		{item, `{"type":"conversation.item.create","previous_item_id":"item_0","item":{"type":"message","role":"user"}}`},
		{RealtimeDeleteItem("item_1"), `{"type":"conversation.item.delete","item_id":"item_1"}`},
		{RealtimeRetrieveItem("item_1"), `{"type":"conversation.item.retrieve","item_id":"item_1"}`},
		{RealtimeTruncateItem("item_1", 0, 1500*time.Millisecond), `{"type":"conversation.item.truncate","item_id":"item_1","content_index":0,"audio_end_ms":1500}`},
		{RealtimeClearInputAudio(), `{"type":"input_audio_buffer.clear"}`},
		{RealtimeMCPApproval("req_1", false, "not allowed"), `{"type":"conversation.item.create","item":{"type":"mcp_approval_response","approval_request_id":"req_1","approve":false,"reason":"not allowed"}}`},
	} {
		assertJSON(t, json.RawMessage(c.got), c.want)
	}
	if _, err := RealtimeCreateItem(map[string]any{"bad": make(chan int)}, ""); err == nil {
		t.Error("an item that cannot encode must fail")
	}
}

func TestMCPTool(t *testing.T) {
	tool := Tools.MCP("docs",
		WithMCPServerURL("https://mcp.example.com"),
		WithMCPAuthorization("token"),
		WithMCPHeaders(map[string]string{"X-Team": "voice"}),
		WithMCPServerDescription("Product docs"),
		WithMCPAllowedTools(MCPToolFilter{ToolNames: []string{"search"}}),
		WithMCPApprovalFilter(MCPToolFilter{ToolNames: []string{"delete"}}, MCPToolFilter{ReadOnly: ptrTo(true)}),
		WithMCPDeferLoading(),
		WithMCPAllowedCallers("direct"),
	)
	if tool.Name != "docs" || tool.ProviderDefinedType != "mcp" {
		t.Fatalf("tool = %+v", tool)
	}
	tools, err := openairealtime.BuildTools([]provider.ToolDefinition{tool})
	if err != nil {
		t.Fatal(err)
	}
	assertJSON(t, tools, `[{"type":"mcp","server_label":"docs","server_url":"https://mcp.example.com","authorization":"token",
		"headers":{"X-Team":"voice"},"server_description":"Product docs","allowed_tools":{"tool_names":["search"]},
		"require_approval":{"always":{"tool_names":["delete"]},"never":{"read_only":true}},"defer_loading":true,"allowed_callers":["direct"]}]`)

	assertJSON(t, Tools.MCP("gmail", WithMCPConnector("connector_gmail"), WithMCPTunnel("tun_1"), WithMCPRequireApproval("never")).ProviderDefinedOptions,
		`{"server_label":"gmail","connector_id":"connector_gmail","tunnel_id":"tun_1","require_approval":"never"}`)
}
