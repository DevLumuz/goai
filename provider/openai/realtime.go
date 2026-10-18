package openai

import (
	"context"
	"errors"
	"net/http"
	"os"

	"github.com/zendev-sh/goai/internal/openairealtime"
	"github.com/zendev-sh/goai/provider"
)

// Realtime creates an OpenAI Realtime model, such as "gpt-realtime-2.1", that
// speaks the GA Realtime API over WebSocket from the backend. It reuses the
// provider options for the API key, base URL, headers, and HTTP client.
//
// Session fields without a common option, such as "reasoning", "truncation",
// or "audio" sub-objects, go in ProviderOptions using their Realtime API names.
// Input transcription defaults to gpt-live-transcribe; set another model with
// TranscriptionConfig.ProviderOptions["model"].
//
// Experimental: see provider.RealtimeModel.
func Realtime(modelID string, opts ...Option) provider.RealtimeModel {
	o := resolveConnectOptions(opts)
	ts := o.tokenSource
	return openairealtime.New(openairealtime.Config{
		Provider:   "openai",
		ModelID:    modelID,
		BaseURL:    o.baseURL,
		Headers:    o.headers,
		HTTPClient: o.httpClient,
		Authorize: func(ctx context.Context, header http.Header) error {
			if ts == nil {
				return errors.New("goai: no API key or token source configured")
			}
			token, err := ts.Token(ctx)
			if err != nil {
				return err
			}
			header.Set("Authorization", "Bearer "+token)
			return nil
		},
	})
}

// resolveConnectOptions applies options and the OPENAI_API_KEY / OPENAI_BASE_URL
// fallbacks shared by the WebSocket models.
func resolveConnectOptions(opts []Option) options {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("OPENAI_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return o
}
