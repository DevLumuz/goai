package azure

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/coder/websocket"
	"github.com/zendev-sh/goai/provider"
	"go.uber.org/goleak"
)

type realtimeHandshake struct{ path, query, auth, apiKey string }

func realtimeServer(t *testing.T) (*httptest.Server, chan realtimeHandshake) {
	t.Helper()
	got := make(chan realtimeHandshake, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		got <- realtimeHandshake{r.URL.Path, r.URL.RawQuery, r.Header.Get("Authorization"), r.Header.Get("api-key")}
		ctx := r.Context()
		if _, _, err := conn.Read(ctx); err != nil {
			return
		}
		_ = conn.Write(ctx, websocket.MessageText, []byte(`{"type":"session.updated"}`))
		for {
			if _, _, err := conn.Read(ctx); err != nil {
				return
			}
		}
	}))
	t.Cleanup(server.Close)
	return server, got
}

func TestRealtime_APIKeyAndEntraID(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	tests := []struct {
		name              string
		opts              []Option
		wantAuth, wantKey string
	}{
		{"api key", []Option{WithAPIKey("az-key")}, "", "az-key"},
		{"entra id", []Option{WithTokenSource(provider.StaticToken("entra-token"))}, "Bearer entra-token", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, got := realtimeServer(t)
			opts := append([]Option{WithEndpoint(server.URL + "/")}, tt.opts...)
			model := Realtime("my-realtime", opts...)
			if model.ModelID() != "my-realtime" {
				t.Fatalf("ModelID = %q", model.ModelID())
			}
			conn, err := model.OpenRealtime(t.Context(), provider.RealtimeParams{})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = conn.Close() }()
			h := <-got
			if h.path != "/openai/v1/realtime" || h.query != "model=my-realtime" {
				t.Errorf("endpoint = %s?%s", h.path, h.query)
			}
			if h.auth != tt.wantAuth || h.apiKey != tt.wantKey {
				t.Errorf("auth = %q, api-key = %q", h.auth, h.apiKey)
			}
		})
	}
}

func TestRealtime_ConfigurationErrors(t *testing.T) {
	for _, env := range []string{"AZURE_OPENAI_ENDPOINT", "AZURE_RESOURCE_NAME", "AZURE_OPENAI_API_KEY", "AZURE_API_KEY"} {
		t.Setenv(env, "")
	}
	if _, err := Realtime("d", WithAPIKey("k")).OpenRealtime(context.Background(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected a missing endpoint error")
	}
	if _, err := Realtime("d", WithEndpoint("https://x.openai.azure.com")).OpenRealtime(context.Background(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected a missing credentials error")
	}
	failing := provider.CachedTokenSource(func(context.Context) (*provider.Token, error) { return nil, context.DeadlineExceeded })
	if _, err := Realtime("d", WithEndpoint("https://x.openai.azure.com"), WithTokenSource(failing)).OpenRealtime(context.Background(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected a token source error")
	}
	t.Setenv("AZURE_RESOURCE_NAME", "res")
	t.Setenv("AZURE_OPENAI_API_KEY", "k")
	if m := Realtime("d"); m.ModelID() != "d" {
		t.Fatal("env-configured model must keep its deployment name")
	}
}

func TestTranslation_Endpoint(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	got := make(chan realtimeHandshake, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		got <- realtimeHandshake{r.URL.Path, r.URL.RawQuery, r.Header.Get("Authorization"), r.Header.Get("api-key")}
		ctx := r.Context()
		_ = conn.Write(ctx, websocket.MessageText, []byte(`{"type":"session.created","session":{"id":"s"}}`))
		for {
			_, data, err := conn.Read(ctx)
			if err != nil {
				return
			}
			if string(data) == `{"type":"session.close"}` {
				_ = conn.Write(ctx, websocket.MessageText, []byte(`{"type":"session.closed"}`))
				return
			}
		}
	}))
	t.Cleanup(server.Close)

	model := Translation("my-translate", WithEndpoint(server.URL), WithAPIKey("az-key"))
	if model.ModelID() != "my-translate" {
		t.Fatalf("ModelID = %q", model.ModelID())
	}
	conn, err := model.OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	h := <-got
	if h.path != "/openai/v1/realtime/translations" || h.query != "model=my-translate" || h.apiKey != "az-key" {
		t.Errorf("handshake = %+v", h)
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
}
