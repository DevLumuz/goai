package google

import (
	"testing"

	"github.com/zendev-sh/goai/provider"
)

// ---------------------------------------------------------------------------
// GoogleSearch
// ---------------------------------------------------------------------------

func TestTools_GoogleSearch_Default(t *testing.T) {
	def := Tools.GoogleSearch()
	if def.Name != "google_search" {
		t.Errorf("Name = %q, want google_search", def.Name)
	}
	if def.ProviderDefinedType != "google.google_search" {
		t.Errorf("ProviderDefinedType = %q, want google.google_search", def.ProviderDefinedType)
	}
	if len(def.ProviderDefinedOptions) != 0 {
		t.Errorf("expected empty options, got %v", def.ProviderDefinedOptions)
	}
}

func TestTools_GoogleSearch_WebSearchOnly(t *testing.T) {
	def := Tools.GoogleSearch(WithWebSearch())
	opts := def.ProviderDefinedOptions
	searchTypes, ok := opts["searchTypes"].(map[string]any)
	if !ok {
		t.Fatal("searchTypes not set")
	}
	if _, ok := searchTypes["webSearch"]; !ok {
		t.Error("webSearch should be set")
	}
	if _, ok := searchTypes["imageSearch"]; ok {
		t.Error("imageSearch should not be set")
	}
}

func TestTools_GoogleSearch_ImageSearchOnly(t *testing.T) {
	def := Tools.GoogleSearch(WithImageSearch())
	opts := def.ProviderDefinedOptions
	searchTypes, ok := opts["searchTypes"].(map[string]any)
	if !ok {
		t.Fatal("searchTypes not set")
	}
	if _, ok := searchTypes["imageSearch"]; !ok {
		t.Error("imageSearch should be set")
	}
}

func TestTools_GoogleSearch_BothSearchTypes(t *testing.T) {
	def := Tools.GoogleSearch(WithWebSearch(), WithImageSearch())
	searchTypes := def.ProviderDefinedOptions["searchTypes"].(map[string]any)
	if _, ok := searchTypes["webSearch"]; !ok {
		t.Error("webSearch missing")
	}
	if _, ok := searchTypes["imageSearch"]; !ok {
		t.Error("imageSearch missing")
	}
}

func TestTools_GoogleSearch_TimeRange(t *testing.T) {
	def := Tools.GoogleSearch(WithTimeRange("2025-01-01T00:00:00Z", "2025-12-31T23:59:59Z"))
	opts := def.ProviderDefinedOptions
	tr, ok := opts["timeRangeFilter"].(map[string]any)
	if !ok {
		t.Fatal("timeRangeFilter not set")
	}
	if tr["startTime"] != "2025-01-01T00:00:00Z" {
		t.Errorf("startTime = %v", tr["startTime"])
	}
	if tr["endTime"] != "2025-12-31T23:59:59Z" {
		t.Errorf("endTime = %v", tr["endTime"])
	}
}

func TestTools_GoogleSearch_AllOptions(t *testing.T) {
	def := Tools.GoogleSearch(
		WithWebSearch(),
		WithImageSearch(),
		WithTimeRange("2025-01-01T00:00:00Z", "2025-06-01T00:00:00Z"),
	)
	opts := def.ProviderDefinedOptions
	if _, ok := opts["searchTypes"]; !ok {
		t.Error("searchTypes not set")
	}
	if _, ok := opts["timeRangeFilter"]; !ok {
		t.Error("timeRangeFilter not set")
	}
}

// ---------------------------------------------------------------------------
// URLContext
// ---------------------------------------------------------------------------

func TestTools_URLContext(t *testing.T) {
	def := Tools.URLContext()
	if def.Name != "url_context" {
		t.Errorf("Name = %q, want url_context", def.Name)
	}
	if def.ProviderDefinedType != "google.url_context" {
		t.Errorf("ProviderDefinedType = %q, want google.url_context", def.ProviderDefinedType)
	}
}

// ---------------------------------------------------------------------------
// CodeExecution
// ---------------------------------------------------------------------------

func TestTools_CodeExecution(t *testing.T) {
	def := Tools.CodeExecution()
	if def.Name != "code_execution" {
		t.Errorf("Name = %q, want code_execution", def.Name)
	}
	if def.ProviderDefinedType != "google.code_execution" {
		t.Errorf("ProviderDefinedType = %q, want google.code_execution", def.ProviderDefinedType)
	}
}

// ---------------------------------------------------------------------------
// googleProviderTool
// ---------------------------------------------------------------------------

func TestGoogleProviderTool_GoogleSearch(t *testing.T) {
	def := Tools.GoogleSearch(WithWebSearch())
	apiTool := googleProviderTool(def)

	// Should map "google.google_search" to camelCase "googleSearch".
	inner, ok := apiTool["googleSearch"]
	if !ok {
		t.Fatal("googleSearch key not found in API tool")
	}
	opts, ok := inner.(map[string]any)
	if !ok {
		t.Fatal("inner should be map")
	}
	if _, ok := opts["searchTypes"]; !ok {
		t.Error("searchTypes should be in API tool options")
	}
}

func TestGoogleProviderTool_URLContext(t *testing.T) {
	def := Tools.URLContext()
	apiTool := googleProviderTool(def)
	if _, ok := apiTool["urlContext"]; !ok {
		t.Fatal("urlContext key not found")
	}
}

func TestGoogleProviderTool_CodeExecution(t *testing.T) {
	def := Tools.CodeExecution()
	apiTool := googleProviderTool(def)
	if _, ok := apiTool["codeExecution"]; !ok {
		t.Fatal("codeExecution key not found")
	}
}

func TestGoogleProviderTool_NoPrefix(t *testing.T) {
	// ProviderDefinedType without "google." prefix should pass through.
	def := provider.ToolDefinition{
		Name:                "custom_tool",
		ProviderDefinedType: "some_tool",
	}
	apiTool := googleProviderTool(def)
	if _, ok := apiTool["someTool"]; !ok {
		t.Fatal("someTool key not found")
	}
}

func TestGoogleProviderTool_WithOptions(t *testing.T) {
	def := provider.ToolDefinition{
		Name:                "test",
		ProviderDefinedType: "google.test_tool",
		ProviderDefinedOptions: map[string]any{
			"key1": "value1",
			"key2": 42,
		},
	}
	apiTool := googleProviderTool(def)
	inner := apiTool["testTool"].(map[string]any)
	if inner["key1"] != "value1" {
		t.Errorf("key1 = %v", inner["key1"])
	}
	if inner["key2"] != 42 {
		t.Errorf("key2 = %v", inner["key2"])
	}
}

// ---------------------------------------------------------------------------
// snakeToCamel
// ---------------------------------------------------------------------------

func TestSnakeToCamel(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"google_search", "googleSearch"},
		{"url_context", "urlContext"},
		{"code_execution", "codeExecution"},
		{"simple", "simple"},
		{"a_b_c", "aBC"},
		{"", ""},
		{"_leading", "Leading"},
	}
	for _, tt := range tests {
		got := snakeToCamel(tt.input)
		if got != tt.want {
			t.Errorf("snakeToCamel(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// ComputerUse
// ---------------------------------------------------------------------------

func TestTools_ComputerUse_Default(t *testing.T) {
	def := Tools.ComputerUse()
	if def.Name != "computer_use" {
		t.Errorf("Name = %q, want computer_use", def.Name)
	}
	if def.ProviderDefinedType != "google.computer_use" {
		t.Errorf("ProviderDefinedType = %q, want google.computer_use", def.ProviderDefinedType)
	}
	if len(def.ProviderDefinedOptions) != 0 {
		t.Errorf("expected empty options, got %v", def.ProviderDefinedOptions)
	}
}

func TestTools_ComputerUse_Options(t *testing.T) {
	def := Tools.ComputerUse(
		WithEnvironment("ENVIRONMENT_BROWSER"),
		WithExcludedFunctions("open_web_browser", "search"),
	)
	opts := def.ProviderDefinedOptions
	if opts["environment"] != "ENVIRONMENT_BROWSER" {
		t.Errorf("environment = %v, want ENVIRONMENT_BROWSER", opts["environment"])
	}
	excluded, ok := opts["excludedPredefinedFunctions"].([]string)
	if !ok {
		t.Fatalf("excludedPredefinedFunctions = %T, want []string", opts["excludedPredefinedFunctions"])
	}
	if len(excluded) != 2 || excluded[0] != "open_web_browser" || excluded[1] != "search" {
		t.Errorf("excludedPredefinedFunctions = %v", excluded)
	}
}

// The tool reaches the wire under its camelCased key, with the options nested
// inside it -- the shape the API expects.
func TestTools_ComputerUse_WireShape(t *testing.T) {
	wire := googleProviderTool(Tools.ComputerUse(WithEnvironment("ENVIRONMENT_BROWSER")))
	inner, ok := wire["computerUse"].(map[string]any)
	if !ok {
		t.Fatalf("wire = %v, want a computerUse key", wire)
	}
	if inner["environment"] != "ENVIRONMENT_BROWSER" {
		t.Errorf("computerUse.environment = %v", inner["environment"])
	}
}
