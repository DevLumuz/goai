package openai

import (
	"context"
	"errors"
	"net/http"

	"github.com/zendev-sh/goai/internal/openairealtime"
	"github.com/zendev-sh/goai/provider"
)

// Translation creates an OpenAI realtime translation model, such as
// "gpt-realtime-translate": it hears speech and speaks it in the target
// language, with optional source and translated transcripts. Set the target
// language with TranslationProviderOptions. Close sends session.close and
// waits for session.closed so the last translated audio is delivered.
//
// Experimental: see provider.RealtimeModel.
func Translation(modelID string, opts ...Option) provider.RealtimeModel {
	o := resolveConnectOptions(opts)
	ts := o.tokenSource
	return openairealtime.NewTranslation(openairealtime.Config{
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

// TranslationConfig holds the translation session settings. Language is the
// target language, such as "es"; NoiseReduction is "near_field" or "far_field".
type TranslationConfig struct {
	Language       string
	NoiseReduction string
}

// TranslationProviderOptions converts cfg to the ProviderOptions of a
// translation session, on OpenAI or Azure OpenAI.
func TranslationProviderOptions(cfg TranslationConfig) map[string]any {
	audio := map[string]any{}
	if cfg.Language != "" {
		audio["output"] = map[string]any{"language": cfg.Language}
	}
	if cfg.NoiseReduction != "" {
		audio["input"] = map[string]any{"noise_reduction": map[string]any{"type": cfg.NoiseReduction}}
	}
	if len(audio) == 0 {
		return map[string]any{}
	}
	return map[string]any{"audio": audio}
}
