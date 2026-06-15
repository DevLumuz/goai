package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestTools_Computer(t *testing.T) {
	tool := Tools.Computer(ComputerToolOptions{
		DisplayWidthPx:  1920,
		DisplayHeightPx: 1080,
		DisplayNumber:   1,
	})

	if tool.Name != "computer" {
		t.Errorf("Name = %q, want computer", tool.Name)
	}
	if tool.ProviderDefinedType != "computer_20250124" {
		t.Errorf("ProviderDefinedType = %q, want computer_20250124", tool.ProviderDefinedType)
	}
	if tool.ProviderDefinedOptions["display_width_px"] != 1920 {
		t.Errorf("display_width_px = %v, want 1920", tool.ProviderDefinedOptions["display_width_px"])
	}
	if tool.ProviderDefinedOptions["display_height_px"] != 1080 {
		t.Errorf("display_height_px = %v, want 1080", tool.ProviderDefinedOptions["display_height_px"])
	}
	if tool.ProviderDefinedOptions["display_number"] != 1 {
		t.Errorf("display_number = %v, want 1", tool.ProviderDefinedOptions["display_number"])
	}
}

func TestTools_ComputerNoDisplayNumber(t *testing.T) {
	tool := Tools.Computer(ComputerToolOptions{
		DisplayWidthPx:  1024,
		DisplayHeightPx: 768,
	})
	if _, ok := tool.ProviderDefinedOptions["display_number"]; ok {
		t.Error("display_number should not be set when 0")
	}
}

func TestTools_Computer20251124(t *testing.T) {
	tool := Tools.Computer_20251124(Computer20251124Options{
		DisplayWidthPx:  1920,
		DisplayHeightPx: 1080,
		EnableZoom:      true,
	})
	if tool.ProviderDefinedType != "computer_20251124" {
		t.Errorf("ProviderDefinedType = %q, want computer_20251124", tool.ProviderDefinedType)
	}
	if tool.ProviderDefinedOptions["enable_zoom"] != true {
		t.Error("enable_zoom should be true")
	}
}

func TestTools_Computer20251124_NoZoom(t *testing.T) {
	tool := Tools.Computer_20251124(Computer20251124Options{
		DisplayWidthPx:  1024,
		DisplayHeightPx: 768,
	})
	if _, ok := tool.ProviderDefinedOptions["enable_zoom"]; ok {
		t.Error("enable_zoom should not be set when false")
	}
}

func TestTools_Bash(t *testing.T) {
	tool := Tools.Bash()
	if tool.Name != "bash" {
		t.Errorf("Name = %q, want bash", tool.Name)
	}
	if tool.ProviderDefinedType != "bash_20250124" {
		t.Errorf("ProviderDefinedType = %q, want bash_20250124", tool.ProviderDefinedType)
	}
}

func TestTools_TextEditor(t *testing.T) {
	tool := Tools.TextEditor()
	if tool.Name != "str_replace_based_edit_tool" {
		t.Errorf("Name = %q, want str_replace_based_edit_tool", tool.Name)
	}
	if tool.ProviderDefinedType != "text_editor_20250429" {
		t.Errorf("ProviderDefinedType = %q, want text_editor_20250429", tool.ProviderDefinedType)
	}
}

func TestTools_TextEditor20250728(t *testing.T) {
	tool := Tools.TextEditor_20250728(WithMaxCharacters(50000))
	if tool.Name != "str_replace_based_edit_tool" {
		t.Errorf("Name = %q, want str_replace_based_edit_tool", tool.Name)
	}
	if tool.ProviderDefinedType != "text_editor_20250728" {
		t.Errorf("ProviderDefinedType = %q, want text_editor_20250728", tool.ProviderDefinedType)
	}
	if tool.ProviderDefinedOptions["max_characters"] != 50000 {
		t.Errorf("max_characters = %v, want 50000", tool.ProviderDefinedOptions["max_characters"])
	}
}

func TestTools_TextEditor20250728_NoOptions(t *testing.T) {
	tool := Tools.TextEditor_20250728()
	if _, ok := tool.ProviderDefinedOptions["max_characters"]; ok {
		t.Error("max_characters should not be set when 0")
	}
}

func TestTools_WebSearch(t *testing.T) {
	tool := Tools.WebSearch(
		WithMaxUses(5),
		WithAllowedDomains("example.com"),
		WithBlockedDomains("spam.com"),
	)
	if tool.Name != "web_search" {
		t.Errorf("Name = %q, want web_search", tool.Name)
	}
	if tool.ProviderDefinedType != "web_search_20250305" {
		t.Errorf("ProviderDefinedType = %q, want web_search_20250305", tool.ProviderDefinedType)
	}
	if tool.ProviderDefinedOptions["max_uses"] != 5 {
		t.Errorf("max_uses = %v, want 5", tool.ProviderDefinedOptions["max_uses"])
	}
	allowed, ok := tool.ProviderDefinedOptions["allowed_domains"].([]string)
	if !ok || len(allowed) != 1 || allowed[0] != "example.com" {
		t.Errorf("allowed_domains = %v, want [example.com]", tool.ProviderDefinedOptions["allowed_domains"])
	}
	blocked, ok := tool.ProviderDefinedOptions["blocked_domains"].([]string)
	if !ok || len(blocked) != 1 || blocked[0] != "spam.com" {
		t.Errorf("blocked_domains = %v, want [spam.com]", tool.ProviderDefinedOptions["blocked_domains"])
	}
}

func TestTools_WebSearch20260209(t *testing.T) {
	tool := Tools.WebSearch_20260209(WithMaxUses(3))
	if tool.ProviderDefinedType != "web_search_20260209" {
		t.Errorf("ProviderDefinedType = %q, want web_search_20260209", tool.ProviderDefinedType)
	}
	if tool.ProviderDefinedOptions["max_uses"] != 3 {
		t.Errorf("max_uses = %v, want 3", tool.ProviderDefinedOptions["max_uses"])
	}
}

func TestTools_WebSearchWithUserLocation(t *testing.T) {
	tool := Tools.WebSearch(WithWebSearchUserLocation(WebSearchLocation{
		Type:     "approximate",
		City:     "San Francisco",
		Region:   "California",
		Country:  "US",
		Timezone: "America/Los_Angeles",
	}))
	loc, ok := tool.ProviderDefinedOptions["user_location"].(map[string]any)
	if !ok {
		t.Fatal("user_location not set")
	}
	if loc["city"] != "San Francisco" {
		t.Errorf("city = %v, want San Francisco", loc["city"])
	}
}

func TestTools_WebFetch(t *testing.T) {
	tool := Tools.WebFetch(
		WithWebFetchMaxUses(10),
		WithWebFetchAllowedDomains("docs.example.com"),
		WithCitations(true),
		WithMaxContentTokens(5000),
	)
	if tool.Name != "web_fetch" {
		t.Errorf("Name = %q, want web_fetch", tool.Name)
	}
	if tool.ProviderDefinedType != "web_fetch_20260209" {
		t.Errorf("ProviderDefinedType = %q, want web_fetch_20260209", tool.ProviderDefinedType)
	}
	if tool.ProviderDefinedOptions["max_uses"] != 10 {
		t.Errorf("max_uses = %v, want 10", tool.ProviderDefinedOptions["max_uses"])
	}
	citations, ok := tool.ProviderDefinedOptions["citations"].(map[string]any)
	if !ok {
		t.Fatal("citations not set")
	}
	if citations["enabled"] != true {
		t.Error("citations.enabled should be true")
	}
	if tool.ProviderDefinedOptions["max_content_tokens"] != 5000 {
		t.Errorf("max_content_tokens = %v, want 5000", tool.ProviderDefinedOptions["max_content_tokens"])
	}
	allowedDomains, ok := tool.ProviderDefinedOptions["allowed_domains"].([]string)
	if !ok || len(allowedDomains) != 1 || allowedDomains[0] != "docs.example.com" {
		t.Errorf("allowed_domains = %v, want [docs.example.com]", tool.ProviderDefinedOptions["allowed_domains"])
	}
}

func TestTools_CodeExecution(t *testing.T) {
	tool := Tools.CodeExecution()
	if tool.Name != "code_execution" {
		t.Errorf("Name = %q, want code_execution", tool.Name)
	}
	if tool.ProviderDefinedType != "code_execution_20260120" {
		t.Errorf("ProviderDefinedType = %q, want code_execution_20260120", tool.ProviderDefinedType)
	}
}

func TestTools_CodeExecution20250825(t *testing.T) {
	tool := Tools.CodeExecution_20250825()
	if tool.ProviderDefinedType != "code_execution_20250825" {
		t.Errorf("ProviderDefinedType = %q, want code_execution_20250825", tool.ProviderDefinedType)
	}
}

func TestTools_ToolSearchToolRegex(t *testing.T) {
	tool := Tools.ToolSearchToolRegex()
	if tool.Name != "tool_search_tool_regex" {
		t.Errorf("Name = %q, want tool_search_tool_regex", tool.Name)
	}
	if tool.ProviderDefinedType != "tool_search_tool_regex_20251119" {
		t.Errorf("ProviderDefinedType = %q, want tool_search_tool_regex_20251119", tool.ProviderDefinedType)
	}
}

func TestTools_ToolSearchToolBM25(t *testing.T) {
	tool := Tools.ToolSearchToolBM25()
	if tool.Name != "tool_search_tool_bm25" {
		t.Errorf("Name = %q, want tool_search_tool_bm25", tool.Name)
	}
	if tool.ProviderDefinedType != "tool_search_tool_bm25_20251119" {
		t.Errorf("ProviderDefinedType = %q, want tool_search_tool_bm25_20251119", tool.ProviderDefinedType)
	}
}

func TestBetaForTool(t *testing.T) {
	tests := []struct {
		toolType string
		want     string
	}{
		{"computer_20241022", "computer-use-2024-10-22"},
		{"bash_20241022", "computer-use-2024-10-22"},
		{"text_editor_20241022", "computer-use-2024-10-22"},
		{"computer_20250124", "computer-use-2025-01-24"},
		{"bash_20250124", "computer-use-2025-01-24"},
		{"text_editor_20250124", "computer-use-2025-01-24"},
		{"text_editor_20250429", "computer-use-2025-01-24"},
		{"computer_20251124", "computer-use-2025-11-24"},
		{"text_editor_20250728", ""},
		{"code_execution_20250825", "code-execution-2025-08-25"},
		{"code_execution_20260120", ""},
		{"web_search_20260209", "code-execution-web-tools-2026-02-09"},
		{"web_fetch_20260209", "code-execution-web-tools-2026-02-09"},
		{"tool_search_tool_regex_20251119", ""}, // GA, no beta header
		{"tool_search_tool_bm25_20251119", ""},  // GA, no beta header
		{"unknown_tool", ""},
		{"", ""},
	}
	for _, tt := range tests {
		got := betaForTool(tt.toolType)
		if got != tt.want {
			t.Errorf("betaForTool(%q) = %q, want %q", tt.toolType, got, tt.want)
		}
	}
}

func TestCollectToolBetas(t *testing.T) {
	tools := []provider.ToolDefinition{
		{Name: "regular", Description: "a regular tool"},
		Tools.Computer(ComputerToolOptions{DisplayWidthPx: 1920, DisplayHeightPx: 1080}),
		Tools.Bash(),
	}

	betas := collectToolBetas(tools)
	if len(betas) != 1 {
		t.Fatalf("len(betas) = %d, want 1 (both 20250124 share same beta)", len(betas))
	}
	if betas[0] != "computer-use-2025-01-24" {
		t.Errorf("beta = %q", betas[0])
	}
}

func TestCollectToolBetas_Mixed(t *testing.T) {
	tools := []provider.ToolDefinition{
		Tools.Computer_20251124(Computer20251124Options{DisplayWidthPx: 800, DisplayHeightPx: 600}),
		Tools.Bash(), // 20250124
	}

	betas := collectToolBetas(tools)
	if len(betas) != 2 {
		t.Fatalf("len(betas) = %d, want 2", len(betas))
	}
}

func TestCollectToolBetas_CodeExecution(t *testing.T) {
	tools := []provider.ToolDefinition{
		Tools.CodeExecution(),          // 20260120 → no beta
		Tools.CodeExecution_20250825(), // 20250825 → beta
	}

	betas := collectToolBetas(tools)
	if len(betas) != 1 {
		t.Fatalf("len(betas) = %d, want 1", len(betas))
	}
	if betas[0] != "code-execution-2025-08-25" {
		t.Errorf("beta = %q", betas[0])
	}
}

func TestCollectToolBetas_WebTools20260209(t *testing.T) {
	tools := []provider.ToolDefinition{
		Tools.WebSearch_20260209(),
		Tools.WebFetch(),
	}

	betas := collectToolBetas(tools)
	if len(betas) != 1 {
		t.Fatalf("len(betas) = %d, want 1 (both share same beta)", len(betas))
	}
	if betas[0] != "code-execution-web-tools-2026-02-09" {
		t.Errorf("beta = %q", betas[0])
	}
}

func TestCollectToolBetas_Empty(t *testing.T) {
	tools := []provider.ToolDefinition{
		{Name: "regular"},
	}
	betas := collectToolBetas(tools)
	if len(betas) != 0 {
		t.Errorf("len(betas) = %d, want 0", len(betas))
	}
}

func TestConvertToolToAPI_ProviderDefined(t *testing.T) {
	tool := Tools.Computer(ComputerToolOptions{
		DisplayWidthPx:  1920,
		DisplayHeightPx: 1080,
	})

	api := convertToolToAPI(tool)

	if api["type"] != "computer_20250124" {
		t.Errorf("type = %v", api["type"])
	}
	if api["name"] != "computer" {
		t.Errorf("name = %v", api["name"])
	}
	if api["display_width_px"] != 1920 {
		t.Errorf("display_width_px = %v", api["display_width_px"])
	}
	// Should NOT have input_schema or description.
	if _, ok := api["input_schema"]; ok {
		t.Error("provider-defined tool should not have input_schema")
	}
}

func TestConvertToolToAPI_Regular(t *testing.T) {
	tool := provider.ToolDefinition{
		Name:        "get_weather",
		Description: "Get the weather",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
	}

	api := convertToolToAPI(tool)

	if _, ok := api["type"]; ok {
		t.Error("regular tool should not have type field")
	}
	if api["name"] != "get_weather" {
		t.Errorf("name = %v", api["name"])
	}
	if api["description"] != "Get the weather" {
		t.Errorf("description = %v", api["description"])
	}
	if api["input_schema"] == nil {
		t.Error("regular tool should have input_schema")
	}
}

func TestConvertToolToAPI_DeferLoading(t *testing.T) {
	tool := provider.ToolDefinition{
		Name:         "get_weather",
		Description:  "Get the weather",
		InputSchema:  json.RawMessage(`{"type":"object"}`),
		DeferLoading: true,
	}

	api := convertToolToAPI(tool)

	if api["defer_loading"] != true {
		t.Errorf("defer_loading = %v, want true", api["defer_loading"])
	}
	if api["name"] != "get_weather" {
		t.Errorf("name = %v, want get_weather", api["name"])
	}
}

func TestConvertToolToAPI_NoDeferLoadingByDefault(t *testing.T) {
	tool := provider.ToolDefinition{
		Name:        "get_weather",
		Description: "Get the weather",
	}

	api := convertToolToAPI(tool)

	// Backward-compatible: absent unless explicitly deferred.
	if _, ok := api["defer_loading"]; ok {
		t.Error("defer_loading should be absent when DeferLoading is false")
	}
}

func TestConvertToolToAPI_RegularNoSchema(t *testing.T) {
	tool := provider.ToolDefinition{
		Name:        "no_schema",
		Description: "A tool without schema",
	}
	api := convertToolToAPI(tool)
	if _, ok := api["input_schema"]; ok {
		t.Error("should not have input_schema when InputSchema is empty")
	}
}

func TestConvertToolToAPI_RegularInvalidSchema(t *testing.T) {
	tool := provider.ToolDefinition{
		Name:        "bad_schema",
		Description: "A tool with invalid schema",
		InputSchema: json.RawMessage(`not json`),
	}
	api := convertToolToAPI(tool)
	// Invalid JSON should not produce input_schema.
	if _, ok := api["input_schema"]; ok {
		t.Error("should not have input_schema when schema is invalid JSON")
	}
}

func TestTools_Computer20251124_DisplayNumber(t *testing.T) {
	tool := Tools.Computer_20251124(Computer20251124Options{
		DisplayWidthPx:  1920,
		DisplayHeightPx: 1080,
		DisplayNumber:   2,
	})
	if tool.ProviderDefinedOptions["display_number"] != 2 {
		t.Errorf("display_number = %v, want 2", tool.ProviderDefinedOptions["display_number"])
	}
}

func TestTools_WebFetch_BlockedDomains(t *testing.T) {
	tool := Tools.WebFetch(
		WithWebFetchBlockedDomains("evil.com", "spam.org"),
	)
	blocked, ok := tool.ProviderDefinedOptions["blocked_domains"].([]string)
	if !ok || len(blocked) != 2 || blocked[0] != "evil.com" {
		t.Errorf("blocked_domains = %v", tool.ProviderDefinedOptions["blocked_domains"])
	}
}

func TestTools_WebFetch_Default(t *testing.T) {
	tool := Tools.WebFetch()
	if tool.Name != "web_fetch" {
		t.Errorf("Name = %q, want web_fetch", tool.Name)
	}
	if tool.ProviderDefinedType != "web_fetch_20260209" {
		t.Errorf("ProviderDefinedType = %q", tool.ProviderDefinedType)
	}
	if len(tool.ProviderDefinedOptions) != 0 {
		t.Errorf("expected empty options, got %v", tool.ProviderDefinedOptions)
	}
}

func TestTools_WebSearch_Default(t *testing.T) {
	tool := Tools.WebSearch()
	if tool.Name != "web_search" {
		t.Errorf("Name = %q, want web_search", tool.Name)
	}
	if tool.ProviderDefinedType != "web_search_20250305" {
		t.Errorf("ProviderDefinedType = %q", tool.ProviderDefinedType)
	}
}

func TestTools_WebSearch_UserLocationPartial(t *testing.T) {
	// Only country set -- other fields should be absent.
	tool := Tools.WebSearch(WithWebSearchUserLocation(WebSearchLocation{
		Country: "US",
	}))
	loc, ok := tool.ProviderDefinedOptions["user_location"].(map[string]any)
	if !ok {
		t.Fatal("user_location not set")
	}
	if loc["country"] != "US" {
		t.Errorf("country = %v", loc["country"])
	}
	if _, ok := loc["city"]; ok {
		t.Error("city should not be set")
	}
	if _, ok := loc["region"]; ok {
		t.Error("region should not be set")
	}
	if _, ok := loc["timezone"]; ok {
		t.Error("timezone should not be set")
	}
}
