package azure

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/zendev-sh/goai/internal/openairealtime"
	"github.com/zendev-sh/goai/provider"
)

// Realtime creates an Azure OpenAI Realtime model for a deployment of a model
// such as gpt-realtime-2.1. It dials the GA endpoint
// wss://{resource}.openai.azure.com/openai/v1/realtime?model={deployment} and
// authenticates with the api-key header (WithAPIKey) or a Microsoft Entra ID
// bearer token (WithTokenSource). The protocol and options match
// openai.Realtime.
//
// Experimental: see provider.RealtimeModel.
func Realtime(deployment string, opts ...Option) provider.RealtimeModel {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}
	resolveOptions(&o)

	apiKey, ts := o.apiKey, o.tokenSource
	return openairealtime.New(openairealtime.Config{
		Provider:   "azure",
		ModelID:    deployment,
		BaseURL:    strings.TrimRight(o.endpoint, "/") + "/openai/v1",
		Headers:    o.headers,
		HTTPClient: o.httpClient,
		Authorize: func(ctx context.Context, header http.Header) error {
			if o.endpoint == "" {
				return errors.New("azure: no endpoint configured (WithEndpoint or AZURE_OPENAI_ENDPOINT)")
			}
			if ts != nil {
				token, err := ts.Token(ctx)
				if err != nil {
					return err
				}
				header.Set("Authorization", "Bearer "+token)
				return nil
			}
			if apiKey == "" {
				return errors.New("azure: no API key or token source configured")
			}
			header.Set("api-key", apiKey)
			return nil
		},
	})
}
