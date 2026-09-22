package bedrock

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

func TestEmbedding_Titan(t *testing.T) {
	wantEmbedding := []float64{0.1, 0.2, 0.3}
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/model/") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if !strings.Contains(r.URL.Path, "/invoke") {
			t.Errorf("embedding should use /invoke endpoint, got %s", r.URL.Path)
		}
		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "AWS4-HMAC-SHA256") {
			t.Errorf("expected SigV4 auth, got %s", auth)
		}

		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)

		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":           wantEmbedding,
			"inputTextTokenCount": 5,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AKIAIOSFODNN7EXAMPLE"),
		WithSecretKey("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
		WithBaseURL(server.URL),
	)

	if model.MaxValuesPerCall() != 1 {
		t.Errorf("MaxValuesPerCall = %d, want 1", model.MaxValuesPerCall())
	}

	result, err := model.DoEmbed(t.Context(), []string{"hello world"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 1 {
		t.Fatalf("len(Embeddings) = %d, want 1", len(result.Embeddings))
	}
	if len(result.Embeddings[0]) != 3 {
		t.Fatalf("embedding length = %d, want 3", len(result.Embeddings[0]))
	}
	for i, v := range wantEmbedding {
		if result.Embeddings[0][i] != v {
			t.Errorf("embedding[%d] = %f, want %f", i, result.Embeddings[0][i], v)
		}
	}
	if result.Usage.InputTokens != 5 {
		t.Errorf("InputTokens = %d, want 5", result.Usage.InputTokens)
	}
	if gotReq["inputText"] != "hello world" {
		t.Errorf("inputText = %v, want 'hello world'", gotReq["inputText"])
	}
	if gotReq["normalize"] != true {
		t.Errorf("normalize = %v, want true", gotReq["normalize"])
	}
}

func TestEmbedding_TitanProviderOptions(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":           []float64{0.1},
			"inputTextTokenCount": 3,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"test"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"dimensions": 512,
			"normalize":  false,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if gotReq["dimensions"] != float64(512) {
		t.Errorf("dimensions = %v, want 512", gotReq["dimensions"])
	}
	if gotReq["normalize"] != false {
		t.Errorf("normalize = %v, want false", gotReq["normalize"])
	}
}

func TestEmbedding_Cohere(t *testing.T) {
	wantEmbeddings := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embeddings": wantEmbeddings,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("cohere.embed-english-v3",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	if model.MaxValuesPerCall() != 96 {
		t.Errorf("MaxValuesPerCall = %d, want 96", model.MaxValuesPerCall())
	}

	result, err := model.DoEmbed(t.Context(), []string{"hello", "world"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"input_type": "search_query",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 2 {
		t.Fatalf("len(Embeddings) = %d, want 2", len(result.Embeddings))
	}
	if gotReq["input_type"] != "search_query" {
		t.Errorf("input_type = %v, want 'search_query'", gotReq["input_type"])
	}
	texts, _ := gotReq["texts"].([]any)
	if len(texts) != 2 {
		t.Errorf("texts length = %d, want 2", len(texts))
	}
}

func TestEmbedding_TitanV1_NoExtraFields(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":           []float64{0.1, 0.2},
			"inputTextTokenCount": 2,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v1",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	// V1 must NOT send normalize or dimensions — they are unsupported.
	if _, ok := gotReq["normalize"]; ok {
		t.Error("titan v1 must not send 'normalize' field")
	}
	if _, ok := gotReq["dimensions"]; ok {
		t.Error("titan v1 must not send 'dimensions' field")
	}
	if gotReq["inputText"] != "hello" {
		t.Errorf("inputText = %v, want 'hello'", gotReq["inputText"])
	}
}

func TestEmbedding_TitanV2_EmbeddingTypes(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":           []float64{0.1},
			"inputTextTokenCount": 1,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"embeddingTypes": []string{"float", "binary"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	types, _ := gotReq["embeddingTypes"].([]any)
	if len(types) != 2 {
		t.Errorf("embeddingTypes = %v, want [float binary]", gotReq["embeddingTypes"])
	}
}

func TestEmbedding_CohereV4Format(t *testing.T) {
	// Cohere v4 returns {"embeddings": {"float": [...]}} instead of {"embeddings": [[...]]}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":{"float":[[0.1,0.2],[0.3,0.4]]}}`))
	}))
	defer server.Close()

	model := Embedding("cohere.embed-v4:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	result, err := model.DoEmbed(t.Context(), []string{"a", "b"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 2 {
		t.Fatalf("len(Embeddings) = %d, want 2", len(result.Embeddings))
	}
	if result.Embeddings[0][0] != 0.1 || result.Embeddings[1][0] != 0.3 {
		t.Errorf("unexpected embeddings: %v", result.Embeddings)
	}
}

func TestEmbedding_TitanMultimodal_TextOnly(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":           []float64{0.1, 0.2},
			"inputTextTokenCount": 3,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-image-v1",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"outputEmbeddingLength": 384},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 1 {
		t.Fatalf("len(Embeddings) = %d, want 1", len(result.Embeddings))
	}
	cfg, _ := gotReq["embeddingConfig"].(map[string]any)
	if cfg == nil || cfg["outputEmbeddingLength"] != float64(384) {
		t.Errorf("embeddingConfig = %v, want outputEmbeddingLength=384", gotReq["embeddingConfig"])
	}
	if gotReq["inputText"] != "hello" {
		t.Errorf("inputText = %v, want 'hello'", gotReq["inputText"])
	}
	// Must not send normalize or dimensions.
	if _, ok := gotReq["normalize"]; ok {
		t.Error("titan-embed-image-v1 must not send 'normalize'")
	}
}

func TestEmbedding_Nova_TextOnly(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embeddings": []map[string]any{
				{"embeddingType": "TEXT", "embedding": []float64{0.5, 0.6, 0.7}},
			},
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.nova-2-multimodal-embeddings-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	result, err := model.DoEmbed(t.Context(), []string{"test text"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"embeddingDimension": 1024,
			"embeddingPurpose":   "TEXT_RETRIEVAL",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 1 || len(result.Embeddings[0]) != 3 {
		t.Fatalf("unexpected embeddings: %v", result.Embeddings)
	}

	params, _ := gotReq["singleEmbeddingParams"].(map[string]any)
	if params == nil {
		t.Fatal("singleEmbeddingParams missing")
	}
	if gotReq["schemaVersion"] != "nova-multimodal-embed-v1" {
		t.Errorf("schemaVersion = %v", gotReq["schemaVersion"])
	}
	if params["embeddingPurpose"] != "TEXT_RETRIEVAL" {
		t.Errorf("embeddingPurpose = %v, want TEXT_RETRIEVAL", params["embeddingPurpose"])
	}
	if params["embeddingDimension"] != float64(1024) {
		t.Errorf("embeddingDimension = %v, want 1024", params["embeddingDimension"])
	}
	textParam, _ := params["text"].(map[string]any)
	if textParam["value"] != "test text" {
		t.Errorf("text.value = %v, want 'test text'", textParam["value"])
	}
}

func TestEmbedding_Marengo27_TextOnly(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":       []float64{0.1, 0.2, 0.3},
			"embeddingOption": "visual-text",
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("twelvelabs.marengo-embed-2-7-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 1 {
		t.Fatalf("len(Embeddings) = %d, want 1", len(result.Embeddings))
	}
	if gotReq["inputType"] != "text" {
		t.Errorf("inputType = %v, want 'text'", gotReq["inputType"])
	}
	if gotReq["inputText"] != "hello" {
		t.Errorf("inputText = %v, want 'hello'", gotReq["inputText"])
	}
	if gotReq["textTruncate"] != "end" {
		t.Errorf("textTruncate = %v, want 'end'", gotReq["textTruncate"])
	}
}

func TestEmbedding_Marengo30_TextOnly(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"data": map[string]any{
				"embedding": []float64{0.4, 0.5},
			},
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("twelvelabs.marengo-embed-3-0-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	result, err := model.DoEmbed(t.Context(), []string{"world"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 1 || len(result.Embeddings[0]) != 2 {
		t.Fatalf("unexpected embeddings: %v", result.Embeddings)
	}
	if gotReq["inputType"] != "text" {
		t.Errorf("inputType = %v, want 'text'", gotReq["inputType"])
	}
	textParam, _ := gotReq["text"].(map[string]any)
	if textParam == nil || textParam["inputText"] != "world" {
		t.Errorf("text.inputText = %v, want 'world'", textParam)
	}
}

func TestEmbedding_Nova_Defaults(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embeddings": []map[string]any{
				{"embeddingType": "TEXT", "embedding": []float64{0.1}},
			},
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.nova-2-multimodal-embeddings-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	params, _ := gotReq["singleEmbeddingParams"].(map[string]any)
	if params["embeddingPurpose"] != "GENERIC_INDEX" {
		t.Errorf("embeddingPurpose = %v, want GENERIC_INDEX", params["embeddingPurpose"])
	}
	if params["embeddingDimension"] != float64(3072) {
		t.Errorf("embeddingDimension = %v, want 3072", params["embeddingDimension"])
	}
	textParam, _ := params["text"].(map[string]any)
	if textParam["truncationMode"] != "END" {
		t.Errorf("truncationMode = %v, want END", textParam["truncationMode"])
	}
}

func TestEmbedding_Nova_DimensionAsFloat64(t *testing.T) {
	// embeddingDimension passed as float64 (e.g. from JSON-decoded options) must be handled.
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embeddings": []map[string]any{{"embedding": []float64{0.1}}},
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.nova-2-multimodal-embeddings-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"embeddingDimension": float64(1024)},
	})
	if err != nil {
		t.Fatal(err)
	}

	params, _ := gotReq["singleEmbeddingParams"].(map[string]any)
	if params["embeddingDimension"] != float64(1024) {
		t.Errorf("embeddingDimension = %v, want 1024", params["embeddingDimension"])
	}
}

func TestEmbedding_Nova_EmptyResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[]}`))
	}))
	defer server.Close()

	model := Embedding("amazon.nova-2-multimodal-embeddings-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for empty nova response")
	}
	if !strings.Contains(err.Error(), "nova returned no embeddings") {
		t.Errorf("error = %v", err)
	}
}

func TestEmbedding_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"message":"invalid model identifier"}`))
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for HTTP 400")
	}
	if !strings.Contains(err.Error(), "invalid model identifier") {
		t.Errorf("error = %v", err)
	}
}

func TestEmbedding_CohereInvalidFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":"unexpected_string"}`))
	}))
	defer server.Close()

	model := Embedding("cohere.embed-english-v3",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for invalid embeddings format")
	}
	if !strings.Contains(err.Error(), "bedrock: unrecognised embeddings format") {
		t.Errorf("error = %v", err)
	}
}

func TestEmbedding_TitanMultimodal_DefaultLength(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{"embedding": []float64{0.1}, "inputTextTokenCount": 1})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-image-v1",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	cfg, _ := gotReq["embeddingConfig"].(map[string]any)
	if cfg == nil || cfg["outputEmbeddingLength"] != float64(1024) {
		t.Errorf("default outputEmbeddingLength = %v, want 1024", cfg)
	}
}

func TestEmbedding_Marengo27_TextTruncate(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{"embedding": []float64{0.1}})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("twelvelabs.marengo-embed-2-7-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"textTruncate": "none"},
	})
	if err != nil {
		t.Fatal(err)
	}

	if gotReq["textTruncate"] != "none" {
		t.Errorf("textTruncate = %v, want 'none'", gotReq["textTruncate"])
	}
}

func TestEmbedding_CohereDefaultInputType(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{"embeddings": [][]float64{{0.1}}})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("cohere.embed-english-v3",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	if gotReq["input_type"] != "search_document" {
		t.Errorf("default input_type = %v, want 'search_document'", gotReq["input_type"])
	}
}

func TestEmbedding_InvalidRegion(t *testing.T) {
	model := &embeddingModel{
		id:   "amazon.titan-embed-text-v2:0",
		opts: options{region: "invalid region!", accessKey: "AK", secretKey: "SK"},
	}
	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for invalid region")
	}
	if !strings.Contains(err.Error(), "invalid AWS region") {
		t.Errorf("error = %v", err)
	}
}

func TestEmbedding_Empty(t *testing.T) {
	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
	)
	result, err := model.DoEmbed(t.Context(), []string{}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 0 {
		t.Errorf("expected empty embeddings, got %d", len(result.Embeddings))
	}
}

func TestEmbedding_MissingCredentials(t *testing.T) {
	model := &embeddingModel{
		id:   "amazon.titan-embed-text-v2:0",
		opts: options{region: "us-east-1"},
	}
	_, err := model.DoEmbed(t.Context(), []string{"test"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for missing credentials")
	}
	if !strings.Contains(err.Error(), "AWS_ACCESS_KEY_ID") {
		t.Errorf("error = %v, want credentials error", err)
	}
}

func TestEmbedding_BearerToken(t *testing.T) {
	var gotAuth string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":           []float64{0.5},
			"inputTextTokenCount": 1,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithBearerToken("my-bearer-token"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	if gotAuth != "Bearer my-bearer-token" {
		t.Errorf("Authorization = %q, want 'Bearer my-bearer-token'", gotAuth)
	}
}

// TestEmbedding_CohereV4_NonFloatEmbeddingType verifies that a response with
// only non-float embeddings (e.g. int8) is parsed successfully rather than
// silently dropping the vectors (fix for #25).
func TestEmbedding_CohereV4_NonFloatEmbeddingType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Response with only int8 embeddings — no "float" key.
		_, _ = w.Write([]byte(`{"id":"x","embeddings":{"int8":[[1,2,3]]},"texts":["hi"]}`))
	}))
	defer server.Close()

	model := Embedding("cohere.embed-v4:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("expected int8 embeddings to be parsed, got error: %v", err)
	}
	if len(result.Embeddings) != 1 || len(result.Embeddings[0]) != 3 {
		t.Fatalf("unexpected embeddings: %v", result.Embeddings)
	}
	if result.Embeddings[0][0] != 1 || result.Embeddings[0][2] != 3 {
		t.Errorf("embedding = %v, want [1 2 3]", result.Embeddings[0])
	}
}

// TestEmbedding_TitanImage_VersionedID verifies that a versioned model ID like
// amazon.titan-embed-image-v1:0 is routed to doTitanMultimodalEmbed, not doTitanEmbed.
func TestEmbedding_TitanImage_VersionedID(t *testing.T) {
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{"embedding": []float64{0.1, 0.2}})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-image-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 {
		t.Fatalf("embeddings len = %d, want 1", len(result.Embeddings))
	}
	// Multimodal format uses embeddingConfig; text-only format uses inputText directly.
	if _, ok := gotBody["embeddingConfig"]; !ok {
		t.Error("expected embeddingConfig key — versioned ID should route to doTitanMultimodalEmbed")
	}
}

// Titan V2 always answers with "embeddingsByType", and every value in it is ONE
// flat vector: "float" a list of floats, "binary" a list of 0/1 integers (one
// per dimension). "embedding" is present too unless embeddingTypes omits
// "float". The payloads below are trimmed copies of live responses from
// amazon.titan-embed-text-v2:0 (eu-west-1).

// TestEmbedding_TitanV2_DefaultResponse is the response every default call
// gets. Parsing embeddingsByType.float as a matrix failed it with
// "json: cannot unmarshal number into Go value of type []float64".
func TestEmbedding_TitanV2_DefaultResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embedding":[-0.094,0.078,0.048],"embeddingsByType":{"float":[-0.094,0.078,0.048]},"inputTextTokenCount":3}`))
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"dimensions": 1024, "normalize": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{-0.094, 0.078, 0.048}
	if len(result.Embeddings) != 1 || len(result.Embeddings[0]) != len(want) {
		t.Fatalf("unexpected embeddings: %v", result.Embeddings)
	}
	for i := range want {
		if result.Embeddings[0][i] != want[i] {
			t.Errorf("vector[%d] = %v, want %v", i, result.Embeddings[0][i], want[i])
		}
	}
	if result.Usage.InputTokens != 3 {
		t.Errorf("InputTokens = %d, want 3", result.Usage.InputTokens)
	}
}

// TestEmbedding_TitanV2_EmbeddingsByType_Binary: embeddingTypes ["binary"]
// drops "embedding" and returns the bits as a flat integer list.
func TestEmbedding_TitanV2_EmbeddingsByType_Binary(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddingsByType":{"binary":[0,1,1,1,1,0]},"inputTextTokenCount":3}`))
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"embeddingTypes": []string{"binary"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{0, 1, 1, 1, 1, 0}
	if len(result.Embeddings) != 1 || len(result.Embeddings[0]) != len(want) {
		t.Fatalf("unexpected embeddings: %v", result.Embeddings)
	}
	for i := range want {
		if result.Embeddings[0][i] != want[i] {
			t.Errorf("vector[%d] = %v, want %v", i, result.Embeddings[0][i], want[i])
		}
	}
	if result.Usage.InputTokens != 3 {
		t.Errorf("InputTokens = %d, want 3", result.Usage.InputTokens)
	}
}

// TestEmbedding_TitanV2_EmbeddingsByType_FloatAndBinary: with both types
// requested the float vector wins.
func TestEmbedding_TitanV2_EmbeddingsByType_FloatAndBinary(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embedding":[0.1,0.2],"embeddingsByType":{"float":[0.1,0.2],"binary":[1,0]},"inputTextTokenCount":1}`))
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"embeddingTypes": []string{"float", "binary"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 || len(result.Embeddings[0]) != 2 || result.Embeddings[0][0] != 0.1 || result.Embeddings[0][1] != 0.2 {
		t.Errorf("unexpected embeddings: %v", result.Embeddings)
	}
}

// TestEmbedding_CohereV4_Int8 covers #25: Cohere v4 int8 embeddings are parsed
// and converted to float64.
func TestEmbedding_CohereV4_Int8(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":{"int8":[[1,-2,3],[4,5,-6]]}}`))
	}))
	defer server.Close()

	model := Embedding("cohere.embed-v4:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"a", "b"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"embedding_types": []string{"int8"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 2 {
		t.Fatalf("len(Embeddings) = %d, want 2", len(result.Embeddings))
	}
	if result.Embeddings[0][0] != 1 || result.Embeddings[0][1] != -2 || result.Embeddings[0][2] != 3 {
		t.Errorf("row0 = %v", result.Embeddings[0])
	}
	if result.Embeddings[1][2] != -6 {
		t.Errorf("row1 = %v", result.Embeddings[1])
	}
}

// TestEmbedding_CohereV4_Uint8 covers #25: uint8 embeddings are parsed.
func TestEmbedding_CohereV4_Uint8(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":{"uint8":[[10,20],[30,40]]}}`))
	}))
	defer server.Close()

	model := Embedding("cohere.embed-v4:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"a", "b"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 2 || result.Embeddings[0][1] != 20 || result.Embeddings[1][0] != 30 {
		t.Errorf("unexpected embeddings: %v", result.Embeddings)
	}
}

// TestEmbedding_CohereV4_Binary covers #25: batched binary embeddings (array of
// base64-packed bit strings) are decoded.
func TestEmbedding_CohereV4_Binary(t *testing.T) {
	v0 := base64.StdEncoding.EncodeToString([]byte{0xC0}) // 11000000
	v1 := base64.StdEncoding.EncodeToString([]byte{0x03}) // 00000011
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embeddings": map[string]any{"binary": []string{v0, v1}},
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("cohere.embed-v4:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"a", "b"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 2 {
		t.Fatalf("len(Embeddings) = %d, want 2", len(result.Embeddings))
	}
	if result.Embeddings[0][0] != 1 || result.Embeddings[0][1] != 1 || result.Embeddings[0][2] != 0 {
		t.Errorf("row0 = %v", result.Embeddings[0])
	}
	if result.Embeddings[1][6] != 1 || result.Embeddings[1][7] != 1 {
		t.Errorf("row1 = %v", result.Embeddings[1])
	}
}

// TestEmbedding_TitanV2_EmbeddingsByType_InvalidFloat: an unparseable
// embeddingsByType value is an error, not an empty vector.
func TestEmbedding_TitanV2_EmbeddingsByType_InvalidFloat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// embeddingsByType present but "float" holds a string, not [...].
		_, _ = w.Write([]byte(`{"embeddingsByType":{"float":"not-an-array"}}`))
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"embeddingTypes": []string{"float"}},
	})
	if err == nil {
		t.Fatal("expected error for invalid float embeddingsByType, got nil")
	}
}

// TestEmbedding_TitanV2_MalformedResponse covers the response-parse error path
// in doTitanEmbed: a body that is not JSON at all.
func TestEmbedding_TitanV2_MalformedResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`not-json`))
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	if _, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{}); err == nil {
		t.Fatal("expected error for malformed Titan response, got nil")
	}
}

// TestParseTitanEmbeddingsByType_NotAnObject covers the top-level unmarshal
// error: embeddingsByType that is not a type-keyed object.
func TestParseTitanEmbeddingsByType_NotAnObject(t *testing.T) {
	_, err := parseTitanEmbeddingsByType(json.RawMessage(`["not-an-object"]`))
	if err == nil {
		t.Fatal("expected error for non-object embeddingsByType, got nil")
	}
}

// TestParseTitanEmbeddingsByType_NoUsableType covers the exhausted-loop error:
// an object carrying neither "float" nor "binary" (and a null value, which is
// skipped rather than returned as an empty vector).
func TestParseTitanEmbeddingsByType_NoUsableType(t *testing.T) {
	for _, raw := range []string{`{"int8":[1,2]}`, `{"float":null}`, `{}`} {
		if _, err := parseTitanEmbeddingsByType(json.RawMessage(raw)); err == nil {
			t.Errorf("%s: expected error, got nil", raw)
		}
	}
}

// TestParseTypedEmbeddings_FloatUnmarshalError covers the "float" case
// json.Unmarshal error (lines 371-373).
func TestParseTypedEmbeddings_FloatUnmarshalError(t *testing.T) {
	_, err := parseTypedEmbeddings(json.RawMessage(`{"float":"not-an-array"}`))
	if err == nil {
		t.Fatal("expected error for malformed float embeddings, got nil")
	}
}

// TestParseTypedEmbeddings_Int8UnmarshalError covers the "int8" case
// json.Unmarshal error (lines 377-379).
func TestParseTypedEmbeddings_Int8UnmarshalError(t *testing.T) {
	_, err := parseTypedEmbeddings(json.RawMessage(`{"int8":"not-an-array"}`))
	if err == nil {
		t.Fatal("expected error for malformed int8 embeddings, got nil")
	}
}

// TestParseTypedEmbeddings_Uint8UnmarshalError covers the "uint8" case
// json.Unmarshal error (lines 383-385).
func TestParseTypedEmbeddings_Uint8UnmarshalError(t *testing.T) {
	_, err := parseTypedEmbeddings(json.RawMessage(`{"uint8":"not-an-array"}`))
	if err == nil {
		t.Fatal("expected error for malformed uint8 embeddings, got nil")
	}
}

// TestParseTypedEmbeddings_NoRecognizedType covers the fall-through error
// (line 391): none of the recognized type keys is present.
func TestParseTypedEmbeddings_NoRecognizedType(t *testing.T) {
	_, err := parseTypedEmbeddings(json.RawMessage(`{"other":[[0.1]]}`))
	if err == nil {
		t.Fatal("expected error when no recognized type key present, got nil")
	}
	if !strings.Contains(err.Error(), "no embeddings in response") {
		t.Errorf("error = %v, want substring 'no embeddings in response'", err)
	}
}

// TestParseBinaryRows_SingleStringDecodeError covers the single-string
// base64 decode failure (lines 413-415 and 437-439): a valid JSON string that
// is not valid base64 makes decodeBinaryVector fail.
func TestParseBinaryRows_SingleStringDecodeError(t *testing.T) {
	_, err := parseBinaryRows(json.RawMessage(`"%%%"`))
	if err == nil {
		t.Fatal("expected error for invalid base64 single string, got nil")
	}
	if !strings.Contains(err.Error(), "decoding binary embedding") {
		t.Errorf("error = %v, want substring 'decoding binary embedding'", err)
	}
}

// TestParseBinaryRows_ArrayUnmarshalError covers the array json.Unmarshal
// error (lines 419-421): the raw value is neither a single string nor an
// array of strings.
func TestParseBinaryRows_ArrayUnmarshalError(t *testing.T) {
	_, err := parseBinaryRows(json.RawMessage(`{"foo":1}`))
	if err == nil {
		t.Fatal("expected error for unrecognised binary format, got nil")
	}
	if !strings.Contains(err.Error(), "unrecognised binary embedding format") {
		t.Errorf("error = %v, want substring 'unrecognised binary embedding format'", err)
	}
}

// TestParseBinaryRows_ArrayElementDecodeError covers the per-element
// decodeBinaryVector error inside the array loop (lines 423-427): an array
// whose first element is valid base64 and whose second is invalid makes the
// loop fail mid-way.
func TestParseBinaryRows_ArrayElementDecodeError(t *testing.T) {
	_, err := parseBinaryRows(json.RawMessage(`["AQI=","%%%"]`))
	if err == nil {
		t.Fatal("expected error for invalid base64 array element, got nil")
	}
	if !strings.Contains(err.Error(), "decoding binary embedding") {
		t.Errorf("error = %v, want substring 'decoding binary embedding'", err)
	}
}

// TestEmbedding_ResponseBodyOverCap verifies the success-path bounded read in
// embed.go: a response body larger than maxEmbedResponseBytes is rejected.
func TestEmbedding_ResponseBodyOverCap(t *testing.T) {
	transport := &fixedBodyTransport{body: io.NopCloser(io.LimitReader(zeroReader{}, int64(maxEmbedResponseBytes+2)))}
	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL("http://fake"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for oversized response body")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("error = %q, want an 'exceeds' over-cap error", err)
	}
}

// TestEmbedding_ErrorBodyBounded verifies the error-path bounded read in
// embed.go: an error response body larger than maxEmbedErrorBytes is truncated,
// so the tail marker never reaches the extracted error message.
func TestEmbedding_ErrorBodyBounded(t *testing.T) {
	const tailMarker = "TAIL-MARKER-SHOULD-NOT-APPEAR"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.Copy(w, io.MultiReader(
			strings.NewReader(`{"message":"`),
			io.LimitReader(zeroReader{}, int64(maxEmbedErrorBytes+len(tailMarker))),
			strings.NewReader(tailMarker),
		))
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error type = %T, want *goai.APIError", err)
	}
	if strings.Contains(apiErr.Message, tailMarker) {
		t.Errorf("error message contains tail marker; error body was not bounded: %q", apiErr.Message)
	}
}
