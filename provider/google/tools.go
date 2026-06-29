package google

import (
	"strings"

	"github.com/zendev-sh/goai/provider"
)

// Tools provides factory functions for Google provider-defined tools.
// These tools use Gemini's built-in capabilities (Google Search grounding,
// URL context, code execution). Requires Gemini 2.0+.
// Matches Vercel AI SDK's google.tools.
var Tools = struct {
	// GoogleSearch enables grounding with Google Search.
	// The model decides when to search based on the prompt.
	// Returns sources via groundingMetadata in response.
	GoogleSearch func(opts ...GoogleSearchOption) provider.ToolDefinition

	// URLContext enables URL context tool that gives Gemini access to web content.
	// The model uses URLs from the prompt to fetch and process content.
	// Requires Gemini 2.0+.
	URLContext func() provider.ToolDefinition

	// CodeExecution enables the model to generate and run Python code.
	// The model can write code, execute it in a sandboxed environment, and
	// use the output to formulate its response.
	// Requires Gemini 2.0+.
	CodeExecution func() provider.ToolDefinition

	// ComputerUse enables Gemini's computer use tool: the model sees screenshots
	// and emits UI actions (click, type, scroll, …) that the client executes,
	// returning a fresh screenshot. Client-executed: each action arrives as a
	// regular functionCall and its result goes back as a functionResponse.
	// The Gemini 3.x family has it built-in (just add the tool); 2.5 needs the
	// dedicated gemini-2.5-computer-use-preview model.
	ComputerUse func(opts ...ComputerUseOption) provider.ToolDefinition
}{
	GoogleSearch:  googleSearchTool,
	URLContext:    urlContextTool,
	CodeExecution: codeExecutionTool,
	ComputerUse:   computerUseTool,
}

// ---------------------------------------------------------------------------
// GoogleSearch
// ---------------------------------------------------------------------------

// GoogleSearchOption configures the Google Search grounding tool.
type GoogleSearchOption func(*googleSearchConfig)

type googleSearchConfig struct {
	// SearchTypes controls which search types to use.
	WebSearch   bool
	ImageSearch bool
	// TimeRangeFilter restricts results to a time range.
	StartTime string // RFC3339 format
	EndTime   string // RFC3339 format
}

// WithWebSearch enables web search results.
func WithWebSearch() GoogleSearchOption {
	return func(c *googleSearchConfig) { c.WebSearch = true }
}

// WithImageSearch enables image search results.
func WithImageSearch() GoogleSearchOption {
	return func(c *googleSearchConfig) { c.ImageSearch = true }
}

// WithTimeRange restricts search results to a specific time range (RFC3339 format).
func WithTimeRange(startTime, endTime string) GoogleSearchOption {
	return func(c *googleSearchConfig) {
		c.StartTime = startTime
		c.EndTime = endTime
	}
}

func googleSearchTool(opts ...GoogleSearchOption) provider.ToolDefinition {
	cfg := &googleSearchConfig{}
	for _, o := range opts {
		o(cfg)
	}

	providerOpts := map[string]any{}

	if cfg.WebSearch || cfg.ImageSearch {
		searchTypes := map[string]any{}
		if cfg.WebSearch {
			searchTypes["webSearch"] = map[string]any{}
		}
		if cfg.ImageSearch {
			searchTypes["imageSearch"] = map[string]any{}
		}
		providerOpts["searchTypes"] = searchTypes
	}

	if cfg.StartTime != "" && cfg.EndTime != "" {
		providerOpts["timeRangeFilter"] = map[string]any{
			"startTime": cfg.StartTime,
			"endTime":   cfg.EndTime,
		}
	}

	return provider.ToolDefinition{
		Name:                   "google_search",
		ProviderDefinedType:    "google.google_search",
		ProviderDefinedOptions: providerOpts,
	}
}

// ---------------------------------------------------------------------------
// URLContext
// ---------------------------------------------------------------------------

func urlContextTool() provider.ToolDefinition {
	return provider.ToolDefinition{
		Name:                "url_context",
		ProviderDefinedType: "google.url_context",
	}
}

// ---------------------------------------------------------------------------
// CodeExecution
// ---------------------------------------------------------------------------

func codeExecutionTool() provider.ToolDefinition {
	return provider.ToolDefinition{
		Name:                "code_execution",
		ProviderDefinedType: "google.code_execution",
	}
}

// ---------------------------------------------------------------------------
// ComputerUse
// ---------------------------------------------------------------------------

// ComputerUseOption configures the computer use tool.
type ComputerUseOption func(*computerUseConfig)

type computerUseConfig struct {
	// Environment is the target surface enum the model controls (e.g. the
	// browser or the full desktop). Passed through verbatim to the wire; the
	// caller owns the exact enum string the API expects.
	Environment string
	// ExcludedPredefinedFunctions lists built-in action names to disable, so the
	// model never emits them (e.g. omit navigation in a kiosk).
	ExcludedPredefinedFunctions []string
}

// WithEnvironment sets the computer use environment (the surface the model
// controls). The value is the API enum string and is sent unchanged.
func WithEnvironment(env string) ComputerUseOption {
	return func(c *computerUseConfig) { c.Environment = env }
}

// WithExcludedFunctions disables specific predefined actions for this session.
func WithExcludedFunctions(fns ...string) ComputerUseOption {
	return func(c *computerUseConfig) { c.ExcludedPredefinedFunctions = fns }
}

func computerUseTool(opts ...ComputerUseOption) provider.ToolDefinition {
	cfg := &computerUseConfig{}
	for _, o := range opts {
		o(cfg)
	}

	providerOpts := map[string]any{}
	if cfg.Environment != "" {
		providerOpts["environment"] = cfg.Environment
	}
	if len(cfg.ExcludedPredefinedFunctions) > 0 {
		providerOpts["excludedPredefinedFunctions"] = cfg.ExcludedPredefinedFunctions
	}

	// googleProviderTool camelCases "google.computer_use" -> "computerUse" and
	// nests providerOpts under it: {"computerUse": {"environment": "..."}}.
	return provider.ToolDefinition{
		Name:                   "computer_use",
		ProviderDefinedType:    "google.computer_use",
		ProviderDefinedOptions: providerOpts,
	}
}

// ---------------------------------------------------------------------------
// API conversion helpers
// ---------------------------------------------------------------------------

// googleProviderTool maps a ProviderDefinedType to the Gemini API tool format.
// Gemini uses camelCase keys: {"googleSearch": {...}}, {"urlContext": {}}, {"codeExecution": {}}.
func googleProviderTool(t provider.ToolDefinition) map[string]any {
	// Map "google.google_search" -> "googleSearch", "google.url_context" -> "urlContext", etc.
	apiKey := t.ProviderDefinedType
	apiKey = strings.TrimPrefix(apiKey, "google.")

	// Special case for Google Search grounding.
	if apiKey == "google_search" {
		apiKey = "googleSearch"
	} else {
		// Convert snake_case to camelCase: "url_context" -> "urlContext"
		apiKey = snakeToCamel(apiKey)
	}

	opts := map[string]any{}
	for k, v := range t.ProviderDefinedOptions {
		opts[k] = v
	}
	return map[string]any{apiKey: opts}
}

func snakeToCamel(s string) string {
	parts := strings.Split(s, "_")
	for i := 1; i < len(parts); i++ {
		if len(parts[i]) > 0 {
			parts[i] = strings.ToUpper(parts[i][:1]) + parts[i][1:]
		}
	}
	return strings.Join(parts, "")
}
