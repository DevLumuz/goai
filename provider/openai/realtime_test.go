package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/coder/websocket"
	"github.com/zendev-sh/goai/provider"
	"go.uber.org/goleak"
)

type realtimeHandshake struct{ path, query, auth string }

// realtimeServer accepts one Realtime session and reports its handshake.
func realtimeServer(t *testing.T) (*httptest.Server, chan realtimeHandshake) {
	t.Helper()
	got := make(chan realtimeHandshake, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		got <- realtimeHandshake{r.URL.Path, r.URL.RawQuery, r.Header.Get("Authorization")}
		ctx := r.Context()
		if _, _, err := conn.Read(ctx); err != nil { // session.update
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

func TestRealtime_EndpointAndAuth(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	server, got := realtimeServer(t)
	model := Realtime("gpt-realtime-2.1", WithAPIKey("sk-test"), WithBaseURL(server.URL+"/v1"))
	if model.ModelID() != "gpt-realtime-2.1" {
		t.Fatalf("ModelID = %q", model.ModelID())
	}
	conn, err := model.OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()

	h := <-got
	if h.path != "/v1/realtime" || h.query != "model=gpt-realtime-2.1" || h.auth != "Bearer sk-test" {
		t.Fatalf("handshake = %+v", h)
	}
}

func TestRealtime_EnvAndMissingKey(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	server, got := realtimeServer(t)
	t.Setenv("OPENAI_API_KEY", "sk-env")
	t.Setenv("OPENAI_BASE_URL", server.URL+"/v1")
	conn, err := Realtime("gpt-realtime-2.1-mini").OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	if h := <-got; h.auth != "Bearer sk-env" {
		t.Fatalf("env key not used: %+v", h)
	}
	_ = conn.Close()

	t.Setenv("OPENAI_API_KEY", "")
	if _, err := Realtime("m").OpenRealtime(context.Background(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected a missing key error")
	}
	failing := provider.CachedTokenSource(func(context.Context) (*provider.Token, error) { return nil, context.DeadlineExceeded })
	if _, err := Realtime("m", WithTokenSource(failing)).OpenRealtime(context.Background(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected a token source error")
	}
}
