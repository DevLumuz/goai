package goai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/google"
	"github.com/zendev-sh/goai/provider/openai"
	"go.uber.org/goleak"
)

// The conformance scenario runs the same public-API flow against every
// realtime adapter, each backed by a minimal local server for its protocol:
// open, stream audio, run the tool the model calls, receive spoken audio and a
// final transcript, finish the response, and close.

type conformanceServer func(t *testing.T, toolResult chan<- string) *httptest.Server

func websocketServer(t *testing.T, handle func(ctx context.Context, conn *websocket.Conn)) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		handle(r.Context(), conn)
	}))
	t.Cleanup(server.Close)
	return server
}

func write(ctx context.Context, conn *websocket.Conn, msg string) {
	_ = conn.Write(ctx, websocket.MessageText, []byte(msg))
}

// readUntil returns the first client message containing marker.
func readUntil(ctx context.Context, conn *websocket.Conn, marker string) (string, bool) {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return "", false
		}
		if strings.Contains(string(data), marker) {
			return string(data), true
		}
	}
}

func geminiLiveServer(t *testing.T, toolResult chan<- string) *httptest.Server {
	return websocketServer(t, func(ctx context.Context, conn *websocket.Conn) {
		if _, ok := readUntil(ctx, conn, `"setup"`); !ok {
			return
		}
		write(ctx, conn, `{"setupComplete":{}}`)
		if _, ok := readUntil(ctx, conn, `"realtimeInput"`); !ok {
			return
		}
		write(ctx, conn, `{"toolCall":{"functionCalls":[{"id":"call-1","name":"lookup","args":{"q":"go"}}]}}`)
		msg, ok := readUntil(ctx, conn, `"toolResponse"`)
		if !ok {
			return
		}
		toolResult <- msg
		write(ctx, conn, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AQID"}}]}}}`)
		write(ctx, conn, `{"serverContent":{"outputTranscription":{"text":"Found it."}}}`)
		write(ctx, conn, `{"serverContent":{"turnComplete":true}}`)
		readUntil(ctx, conn, "never")
	})
}

func openAIRealtimeServer(t *testing.T, toolResult chan<- string) *httptest.Server {
	return websocketServer(t, func(ctx context.Context, conn *websocket.Conn) {
		write(ctx, conn, `{"type":"session.created"}`)
		if _, ok := readUntil(ctx, conn, `"session.update"`); !ok {
			return
		}
		write(ctx, conn, `{"type":"session.updated"}`)
		if _, ok := readUntil(ctx, conn, `"input_audio_buffer.append"`); !ok {
			return
		}
		write(ctx, conn, `{"type":"response.created","response":{"id":"r1"}}`)
		write(ctx, conn, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call-1","name":"lookup","arguments":"{\"q\":\"go\"}"}`)
		write(ctx, conn, `{"type":"response.done","response":{"id":"r1","status":"completed","output":[{"type":"function_call"}]}}`)
		msg, ok := readUntil(ctx, conn, `"function_call_output"`)
		if !ok {
			return
		}
		toolResult <- msg
		if _, ok := readUntil(ctx, conn, `"response.create"`); !ok {
			return
		}
		write(ctx, conn, `{"type":"response.created","response":{"id":"r2"}}`)
		write(ctx, conn, `{"type":"response.output_audio.delta","response_id":"r2","item_id":"i2","delta":"AQID"}`)
		write(ctx, conn, `{"type":"response.output_audio_transcript.done","response_id":"r2","item_id":"i2","transcript":"Found it."}`)
		write(ctx, conn, `{"type":"response.done","response":{"id":"r2","status":"completed","output":[{"type":"message"}]}}`)
		readUntil(ctx, conn, "never")
	})
}

func gptLiveServer(t *testing.T, toolResult chan<- string) *httptest.Server {
	return websocketServer(t, func(ctx context.Context, conn *websocket.Conn) {
		if _, ok := readUntil(ctx, conn, `"session.start"`); !ok {
			return
		}
		write(ctx, conn, `{"type":"session.started","session":{"id":"s"}}`)
		if _, ok := readUntil(ctx, conn, `"session.input_audio.append"`); !ok {
			return
		}
		write(ctx, conn, `{"type":"session.delegation.created","offset_ms":0,"delegation":{"id":"d1","target":"responses","response_id":"r1"}}`)
		write(ctx, conn, `{"type":"response.event","delegation_id":"d1","event":{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call-1","name":"lookup","arguments":"{\"q\":\"go\"}"}}}`)
		msg, ok := readUntil(ctx, conn, `"function_call_output"`)
		if !ok {
			return
		}
		toolResult <- msg
		if _, ok := readUntil(ctx, conn, `"response.create"`); !ok {
			return
		}
		write(ctx, conn, `{"type":"session.output_audio.delta","delta":"AQID"}`)
		write(ctx, conn, `{"type":"session.output_transcript.delta","delta":"Found it."}`)
		if _, ok := readUntil(ctx, conn, `"session.close"`); ok {
			write(ctx, conn, `{"type":"session.closed","usage":{"seconds":3}}`)
		}
		readUntil(ctx, conn, "never")
	})
}

func TestRealtimeConformance(t *testing.T) {
	adapters := map[string]struct {
		server conformanceServer
		model  func(baseURL string) provider.RealtimeModel
		opts   []goai.RealtimeOption
		frame  int // bytes in 100 ms of input audio
		// turns reports whether the model signals the end of each response
		// with RealtimeResponseFinished and a final transcript. GPT-Live is
		// full duplex: it streams transcript deltas and never ends a turn.
		turns bool
	}{
		"gemini-live": {
			server: geminiLiveServer,
			model: func(base string) provider.RealtimeModel {
				return google.Live("gemini-3.8-live", google.WithAPIKey("k"), google.WithBaseURL(base))
			},
			frame: 3200,
			turns: true,
		},
		"openai-realtime": {
			server: openAIRealtimeServer,
			model: func(base string) provider.RealtimeModel {
				return openai.Realtime("gpt-realtime-2.1", openai.WithAPIKey("k"), openai.WithBaseURL(base+"/v1"))
			},
			frame: 4800,
			turns: true,
		},
		"openai-gpt-live": {
			server: gptLiveServer,
			model: func(base string) provider.RealtimeModel {
				return openai.Live("gpt-live-1", openai.WithAPIKey("k"), openai.WithBaseURL(base+"/v1"))
			},
			opts:  []goai.RealtimeOption{goai.WithRealtimeProviderOptions(openai.LiveResponsesDelegation("gpt-6-luna"))},
			frame: 4800,
		},
	}

	for name, adapter := range adapters {
		t.Run(name, func(t *testing.T) {
			ignore := goleak.IgnoreCurrent()
			t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

			toolResult := make(chan string, 1)
			server := adapter.server(t, toolResult)

			var toolInput string
			lookup := goai.Tool{
				Name:        "lookup",
				Description: "Looks something up.",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
				Execute: func(_ context.Context, input json.RawMessage) (string, error) {
					toolInput = string(input)
					return `{"answer":42}`, nil
				},
			}
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			opts := append([]goai.RealtimeOption{
				goai.WithRealtimeTools(lookup),
				goai.WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true}),
			}, adapter.opts...)
			session, err := goai.OpenRealtimeSession(ctx, adapter.model(server.URL), opts...)
			if err != nil {
				t.Fatal(err)
			}

			if err := session.InputAudio(ctx, make([]byte, adapter.frame)); err != nil {
				t.Fatal(err)
			}

			var (
				sawReady, sawToolCall bool
				audio                 []byte
				transcript            string
			)
		loop:
			for {
				select {
				case ev, ok := <-session.Events():
					if !ok {
						t.Fatalf("events closed early: %v", session.Err())
					}
					switch ev.Type {
					case provider.RealtimeSessionReady:
						sawReady = true
					case provider.RealtimeToolCallEvent:
						sawToolCall = ev.ToolCall != nil && ev.ToolCall.Name == "lookup"
					case provider.RealtimeOutputAudio:
						audio = append(audio, ev.Audio...)
						if ev.AudioFormat.SampleRate != 24_000 {
							t.Errorf("output audio format = %+v", ev.AudioFormat)
						}
					case provider.RealtimeOutputTranscript:
						if ev.IsFinal {
							transcript = ev.Text
						} else if !adapter.turns {
							transcript += ev.Text
							if transcript == "Found it." {
								break loop
							}
						}
					case provider.RealtimeResponseFinished:
						if ev.FinishReason == provider.FinishStop {
							break loop
						}
					}
				case <-ctx.Done():
					t.Fatal("timed out before the spoken response finished")
				}
			}

			select {
			case msg := <-toolResult:
				if !strings.Contains(msg, "call-1") || !strings.Contains(msg, "42") {
					t.Errorf("tool result on the wire = %s", msg)
				}
			default:
				t.Error("the tool result never reached the provider")
			}
			if !sawReady || !sawToolCall || toolInput != `{"q":"go"}` {
				t.Errorf("ready=%v toolCall=%v toolInput=%q", sawReady, sawToolCall, toolInput)
			}
			if string(audio) != "\x01\x02\x03" || transcript != "Found it." {
				t.Errorf("audio=%v transcript=%q", audio, transcript)
			}
			if err := session.Close(); err != nil {
				t.Fatalf("Close: %v", err)
			}
			if err := session.SendText(ctx, "late"); err == nil {
				t.Error("sending after Close must fail")
			}
		})
	}
}
