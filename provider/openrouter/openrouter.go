// Package openrouter provides an OpenRouter language model implementation for GoAI.
//
// OpenRouter is a unified API for accessing multiple LLM providers through
// a single OpenAI-compatible endpoint.
package openrouter

import (
	"net/http"
	"os"

	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

const defaultBaseURL = "https://openrouter.ai/api/v1"

// Option configures the OpenRouter provider.
type Option func(*options)

type options struct {
	tokenSource            provider.TokenSource
	baseURL                string
	headers                map[string]string
	httpClient             *http.Client
	providerRouting        []string
	route                  string
	sessionID              string
	useMaxCompletionTokens *bool
}

// WithAPIKey sets a static API key for authentication.
func WithAPIKey(key string) Option {
	return func(o *options) { o.tokenSource = provider.StaticToken(key) }
}

// WithTokenSource sets a dynamic token source for authentication.
func WithTokenSource(ts provider.TokenSource) Option {
	return func(o *options) { o.tokenSource = ts }
}

// WithBaseURL overrides the default API base URL.
func WithBaseURL(url string) Option {
	return func(o *options) { o.baseURL = url }
}

// WithHeaders sets additional HTTP headers sent with every request.
func WithHeaders(h map[string]string) Option {
	return func(o *options) { o.headers = h }
}

// WithHTTPClient sets a custom HTTP client for all requests.
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) { o.httpClient = c }
}

// WithProviderRouting sets OpenRouter provider routing preferences. Each entry
// is a provider ID (or "Auto" for any). They are sent as the body `provider`
// field with the given order and fallbacks disabled, e.g.
// {"order":["Anthropic","Auto"],"allow_fallbacks":false}.
func WithProviderRouting(prefs ...string) Option {
	return func(o *options) { o.providerRouting = prefs }
}

// WithRoute pins the request to a specific OpenRouter route (e.g. "fallback").
// Sent as the body `route` field.
func WithRoute(route string) Option {
	return func(o *options) { o.route = route }
}

// WithSessionID attaches a session identifier to the request for OpenRouter's
// session-based pricing and analytics. Sent as the body `session_id` field.
func WithSessionID(id string) Option {
	return func(o *options) { o.sessionID = id }
}

// WithUseMaxCompletionTokens forces the max_tokens / max_completion_tokens
// choice for every model routed through OpenRouter. OpenRouter fronts many
// upstream providers, some of which only accept max_tokens, so this is
// opt-in: leave unset to keep the model-id heuristic (IsReasoningModel).
func WithUseMaxCompletionTokens(use bool) Option {
	return func(o *options) { o.useMaxCompletionTokens = &use }
}

// Chat creates an OpenRouter language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("OPENROUTER_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("OPENROUTER_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return openaicompat.NewChatModel(openaicompat.ChatModelConfig{
		ProviderID:        "openrouter",
		ModelID:           modelID,
		BaseURL:           o.baseURL,
		TokenSource:       o.tokenSource,
		TokenRequired:     true,
		Headers:           openaicompat.MergeHeaders(o.headers, openRouterFixedHeaders),
		HTTPClient:        o.httpClient,
		Capabilities:      chatCaps,
		WarnPromptCaching: true,
		RequestConfig: openaicompat.RequestConfig{
			IncludeStreamOptions: true,
			// OpenRouter is a router to many upstreams; some only accept
			// max_tokens, so max_completion_tokens is NOT forced for every
			// model. Callers opt in via WithUseMaxCompletionTokens.
			UseMaxCompletionTokens: o.useMaxCompletionTokens,
			ExtraBody:              buildExtraBody(o),
		},
	})
}

// openRouterFixedHeaders are the analytics headers OpenRouter recommends
// sending on every request (https://openrouter.ai/docs/api-reference/overview#headers).
var openRouterFixedHeaders = map[string]string{
	"HTTP-Referer": "https://github.com/zendev-sh/goai",
	"X-Title":      "goai",
}

// buildExtraBody assembles the OpenRouter-specific request body fields:
// the always-on usage include plus the optional provider routing, route and
// session_id keys (item 51).
func buildExtraBody(o options) map[string]any {
	body := map[string]any{"usage": map[string]any{"include": true}}
	if len(o.providerRouting) > 0 {
		body["provider"] = map[string]any{
			"order":           o.providerRouting,
			"allow_fallbacks": false,
		}
	}
	if o.route != "" {
		body["route"] = o.route
	}
	if o.sessionID != "" {
		body["session_id"] = o.sessionID
	}
	return body
}

var chatCaps = provider.ModelCapabilities{
	Temperature:      true,
	ToolCall:         true,
	InputModalities:  provider.ModalitySet{Text: true},
	OutputModalities: provider.ModalitySet{Text: true},
}
