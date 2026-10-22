package openai

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/coder/websocket"
	"github.com/zendev-sh/goai/provider"
)

func TestTranslationEndpointAndCredentials(t *testing.T) {
	liveNoLeaks(t)

	requests := make(chan *http.Request, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests <- r.Clone(context.Background())
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		liveWrite(r.Context(), conn, `{"type":"session.created","session":{"id":"s"}}`)
		for {
			ev, _, ok := liveRead(r.Context(), conn)
			if !ok {
				return
			}
			if ev["type"] == "session.close" {
				liveWrite(r.Context(), conn, `{"type":"session.closed"}`)
				return
			}
		}
	}))
	defer server.Close()

	conn, err := Translation("gpt-realtime-translate", WithAPIKey("k"), WithBaseURL(server.URL+"/v1")).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	r := <-requests
	if r.URL.Path != "/v1/realtime/translations" || r.URL.Query().Get("model") != "gpt-realtime-translate" || r.Header.Get("Authorization") != "Bearer k" {
		t.Errorf("request = %s %v", r.URL, r.Header)
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OPENAI_API_KEY", "")
	if _, err := Translation("m", WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
		t.Error("missing credentials must fail")
	}
}

func TestTranslationProviderOptions(t *testing.T) {
	assertJSON(t, TranslationProviderOptions(TranslationConfig{Language: "es", NoiseReduction: "far_field"}),
		`{"audio":{"output":{"language":"es"},"input":{"noise_reduction":{"type":"far_field"}}}}`)
	assertJSON(t, TranslationProviderOptions(TranslationConfig{}), `{}`)
}

func TestResumptionUnsupportedOnOpenAI(t *testing.T) {
	params := provider.RealtimeParams{Resumption: &provider.RealtimeResumption{}}
	if _, err := Translation("m", WithAPIKey("k")).OpenRealtime(t.Context(), params); !errors.Is(err, provider.ErrRealtimeUnsupported) {
		t.Errorf("translation: %v", err)
	}
	if err := validateLiveParams(params); !errors.Is(err, provider.ErrRealtimeUnsupported) {
		t.Errorf("GPT-Live: %v", err)
	}
	if _, err := Realtime("m", WithAPIKey("k")).OpenRealtime(t.Context(), params); !errors.Is(err, provider.ErrRealtimeUnsupported) {
		t.Errorf("Realtime: %v", err)
	}
}
