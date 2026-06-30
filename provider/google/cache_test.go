package google

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// findToolEntry returns the tools[] entry that has the given key, or nil.
func findToolEntry(t *testing.T, body map[string]any, key string) map[string]any {
	t.Helper()
	tools, ok := body["tools"].([]any)
	if !ok {
		t.Fatalf("body.tools is not an array: %v", body["tools"])
	}
	for _, e := range tools {
		m, ok := e.(map[string]any)
		if !ok {
			continue
		}
		if _, has := m[key]; has {
			return m
		}
	}
	return nil
}

func TestCacheClient_Create_DirectShape(t *testing.T) {
	var gotMethod, gotPath, gotKey string
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod, gotPath, gotKey = r.Method, r.URL.Path, r.Header.Get("x-goog-api-key")
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"name":"cachedContents/abc","expireTime":"2026-01-01T00:00:00Z"}`)
	}))
	defer server.Close()

	c := NewCacheClient(WithAPIKey("test-key"), WithBaseURL(server.URL))
	got, err := c.Create(context.Background(), CachedContentInput{
		Model:  "gemini-3.5-flash",
		System: "you are helpful",
		Tools: []provider.ToolDefinition{
			{Name: "read_file", Description: "reads a file", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		TTL: 5 * time.Minute,
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	if got.Name != "cachedContents/abc" {
		t.Errorf("Name = %q", got.Name)
	}
	if got.ExpiresAt.IsZero() {
		t.Error("ExpiresAt not parsed")
	}
	if gotMethod != http.MethodPost {
		t.Errorf("method = %q", gotMethod)
	}
	if gotPath != "/v1beta/cachedContents" {
		t.Errorf("path = %q", gotPath)
	}
	if gotKey != "test-key" {
		t.Errorf("x-goog-api-key = %q", gotKey)
	}
	if body["model"] != "models/gemini-3.5-flash" {
		t.Errorf("model = %v", body["model"])
	}
	if body["ttl"] != "300s" {
		t.Errorf("ttl = %v", body["ttl"])
	}
	if _, ok := body["systemInstruction"]; !ok {
		t.Error("systemInstruction missing from cache body")
	}
	if findToolEntry(t, body, "functionDeclarations") == nil {
		t.Error("functionDeclarations missing from cache body")
	}
}

// The parity test: a cached resource must hold the SAME server tools and
// toolConfig that a generate request would send — otherwise a cached turn loses
// web_search/url_context and the mixed-tools config.
func TestCacheClient_Create_IncludesServerToolsAndToolConfig(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"name":"cachedContents/x","expireTime":"2026-01-01T00:00:00Z"}`)
	}))
	defer server.Close()

	c := NewCacheClient(WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := c.Create(context.Background(), CachedContentInput{
		Model:  "gemini-3.5-flash",
		System: "sys",
		Tools: []provider.ToolDefinition{
			{Name: "read_file", Description: "reads", InputSchema: json.RawMessage(`{"type":"object"}`)},
			{ProviderDefinedType: "google.google_search"},
			{ProviderDefinedType: "google.url_context"},
		},
		ProviderOptions: map[string]any{
			"toolConfig": map[string]any{
				"includeServerSideToolInvocations": true,
				"functionCallingConfig":            map[string]any{"mode": "VALIDATED"},
			},
		},
		TTL: time.Hour,
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	if findToolEntry(t, body, "functionDeclarations") == nil {
		t.Error("client function declarations missing")
	}
	if findToolEntry(t, body, "googleSearch") == nil {
		t.Error("server tool googleSearch missing from cache body")
	}
	if findToolEntry(t, body, "urlContext") == nil {
		t.Error("server tool urlContext missing from cache body")
	}
	tc, ok := body["toolConfig"].(map[string]any)
	if !ok {
		t.Fatalf("toolConfig missing/!object: %v", body["toolConfig"])
	}
	if tc["includeServerSideToolInvocations"] != true {
		t.Errorf("includeServerSideToolInvocations = %v", tc["includeServerSideToolInvocations"])
	}
}

func TestCacheClient_Renew_PatchesTTL(t *testing.T) {
	var gotMethod, gotPath, gotQuery string
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod, gotPath, gotQuery = r.Method, r.URL.Path, r.URL.RawQuery
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"name":"cachedContents/abc","expireTime":"2026-01-01T01:00:00Z"}`)
	}))
	defer server.Close()

	c := NewCacheClient(WithAPIKey("k"), WithBaseURL(server.URL))
	got, err := c.Renew(context.Background(), "cachedContents/abc", time.Hour)
	if err != nil {
		t.Fatalf("Renew: %v", err)
	}
	if got.ExpiresAt.IsZero() {
		t.Error("ExpiresAt not parsed")
	}
	if gotMethod != http.MethodPatch {
		t.Errorf("method = %q", gotMethod)
	}
	if gotPath != "/v1beta/cachedContents/abc" {
		t.Errorf("path = %q", gotPath)
	}
	if gotQuery != "updateMask=ttl" {
		t.Errorf("query = %q", gotQuery)
	}
	if body["ttl"] != "3600s" {
		t.Errorf("ttl = %v", body["ttl"])
	}
}

func TestCacheClient_Create_VertexShape(t *testing.T) {
	var gotPath, gotAuth string
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotAuth = r.URL.Path, r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"name":"projects/proj/locations/us-central1/cachedContents/z","expireTime":"2026-01-01T00:00:00Z"}`)
	}))
	defer server.Close()

	c := NewCacheClient(
		WithTokenSource(provider.StaticToken("tok")),
		WithVertex("proj", "us-central1"),
		WithBaseURL(server.URL),
	)
	got, err := c.Create(context.Background(), CachedContentInput{Model: "gemini-3.5-flash", System: "x", TTL: time.Hour})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if got.Name != "projects/proj/locations/us-central1/cachedContents/z" {
		t.Errorf("Name = %q", got.Name)
	}
	if gotPath != "/v1beta1/projects/proj/locations/us-central1/cachedContents" {
		t.Errorf("path = %q", gotPath)
	}
	if gotAuth != "Bearer tok" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	if body["model"] != "projects/proj/locations/us-central1/publishers/google/models/gemini-3.5-flash" {
		t.Errorf("model = %v", body["model"])
	}
}

func TestCacheClient_Create_SurfacesError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":{"code":400,"message":"bad"}}`)
	}))
	defer server.Close()

	c := NewCacheClient(WithAPIKey("k"), WithBaseURL(server.URL))
	if _, err := c.Create(context.Background(), CachedContentInput{Model: "m", TTL: time.Minute}); err == nil {
		t.Error("expected error on 400")
	}
}
