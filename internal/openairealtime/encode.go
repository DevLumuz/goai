package openairealtime

import (
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/zendev-sh/goai/internal/jsonmerge"
	"github.com/zendev-sh/goai/provider"
)

// encode maps a client event to wire events. response, when set, is the
// response the event asks for; it goes through the sequencer.
func (c *conn) encode(event provider.RealtimeClientEvent) (msgs []clientEvent, response *createRequest, err error) {
	auto := &createRequest{}
	switch event.Type {
	case provider.RealtimeInputAudioAppend:
		if len(event.Audio) == 0 {
			return nil, nil, fmt.Errorf("%w: empty audio frame", provider.ErrInvalidRealtimeConfig)
		}
		return []clientEvent{{"type": "input_audio_buffer.append", "audio": base64.StdEncoding.EncodeToString(event.Audio)}}, nil, nil
	case provider.RealtimeInputAudioCommit:
		return []clientEvent{{"type": "input_audio_buffer.commit"}}, nil, nil
	case provider.RealtimeResponseCreate:
		return nil, &createRequest{params: event.ProviderOptions}, nil
	case provider.RealtimeTextInput:
		msg := itemCreate(map[string]any{
			"type":    "message",
			"role":    "user",
			"content": []map[string]any{{"type": "input_text", "text": event.Text}},
		})
		if c.autoResponse {
			return []clientEvent{msg}, auto, nil
		}
		return []clientEvent{msg}, nil, nil
	case provider.RealtimeImageInput:
		media := event.Media
		if media == nil || !strings.HasPrefix(media.MIMEType, "image/") || len(media.Data) == 0 {
			return nil, nil, fmt.Errorf("%w: image input needs an image MIME type and data", provider.ErrInvalidRealtimeConfig)
		}
		return []clientEvent{itemCreate(map[string]any{
			"type":    "message",
			"role":    "user",
			"content": []map[string]any{{"type": "input_image", "image_url": dataURL(media.MIMEType, media.Data)}},
		})}, nil, nil
	case provider.RealtimeConversationAppend:
		conv := event.Conversation
		if conv == nil || len(conv.Messages) == 0 {
			return nil, nil, fmt.Errorf("%w: conversation needs messages", provider.ErrInvalidRealtimeConfig)
		}
		items, err := ConversationItems(conv.Messages)
		if err != nil {
			return nil, nil, err
		}
		for _, item := range items {
			msgs = append(msgs, itemCreate(item))
		}
		if conv.RequestResponse {
			return msgs, auto, nil
		}
		return msgs, nil, nil
	case provider.RealtimeSessionUpdateEvent:
		msg, err := encodeSessionUpdate(event.Update)
		return []clientEvent{msg}, nil, err
	case provider.RealtimeToolResultEvent:
		item, err := toolOutputItem(event.ToolResult)
		if err != nil {
			return nil, nil, err
		}
		return []clientEvent{itemCreate(item)}, nil, nil
	case provider.RealtimeResponseCancel:
		return c.encodeInterrupt(event.Cursor), nil, nil
	default:
		return nil, nil, fmt.Errorf("%w: openai realtime does not support %s", provider.ErrRealtimeUnsupported, event.Type)
	}
}

func itemCreate(item map[string]any) clientEvent {
	return clientEvent{"type": "conversation.item.create", "item": item}
}

func dataURL(mimeType string, data []byte) string {
	return "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data)
}

// toolOutputItem builds a function_call_output. Its output is a string, so
// partial results and media have no place in it.
func toolOutputItem(result *provider.RealtimeToolResult) (map[string]any, error) {
	if result == nil || result.ToolCallID == "" {
		return nil, fmt.Errorf("%w: tool result needs a tool call ID", provider.ErrInvalidRealtimeConfig)
	}
	if result.Partial || len(result.Media) > 0 {
		return nil, fmt.Errorf("%w: openai realtime tool outputs are a single text result", provider.ErrRealtimeUnsupported)
	}
	item := map[string]any{
		"type":    "function_call_output",
		"call_id": result.ToolCallID,
		"output":  string(result.Output),
	}
	for key := range result.ProviderOptions {
		if key == "type" || key == "call_id" {
			return nil, fmt.Errorf("%w: %s is derived from the tool result", provider.ErrInvalidRealtimeConfig, key)
		}
	}
	return jsonmerge.Deep(item, result.ProviderOptions), nil
}

// ConversationItems maps messages to conversation items: one message item per
// role turn, plus one item per function call and function output.
func ConversationItems(messages []provider.Message) ([]map[string]any, error) {
	var items []map[string]any
	for _, m := range messages {
		var content []map[string]any
		flush := func(role string) {
			if len(content) > 0 {
				items = append(items, map[string]any{"type": "message", "role": role, "content": content})
				content = nil
			}
		}
		role := string(m.Role)
		for _, part := range m.Content {
			switch {
			case part.Type == provider.PartText && m.Role == provider.RoleAssistant:
				content = append(content, map[string]any{"type": "output_text", "text": part.Text})
			case part.Type == provider.PartText && (m.Role == provider.RoleUser || m.Role == provider.RoleSystem):
				content = append(content, map[string]any{"type": "input_text", "text": part.Text})
			case part.Type == provider.PartImage && m.Role == provider.RoleUser && part.URL != "":
				image := map[string]any{"type": "input_image", "image_url": part.URL}
				if part.Detail != "" {
					image["detail"] = part.Detail
				}
				content = append(content, image)
			case part.Type == provider.PartToolCall && m.Role == provider.RoleAssistant:
				flush(role)
				args := string(part.ToolInput)
				if args == "" {
					args = "{}"
				}
				items = append(items, map[string]any{"type": "function_call", "call_id": part.ToolCallID, "name": part.ToolName, "arguments": args})
			case part.Type == provider.PartToolResult:
				flush(role)
				items = append(items, map[string]any{"type": "function_call_output", "call_id": part.ToolCallID, "output": part.ToolOutput})
			default:
				return nil, fmt.Errorf("%w: openai realtime cannot hold a %s part in a %s message", provider.ErrInvalidRealtimeConfig, part.Type, m.Role)
			}
		}
		flush(role)
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("%w: conversation has no content openai realtime accepts", provider.ErrInvalidRealtimeConfig)
	}
	return items, nil
}

// encodeSessionUpdate maps a session update to session.update. Nil fields stay
// out so the server keeps them; an empty tool list clears the tools.
func encodeSessionUpdate(update *provider.RealtimeSessionUpdate) (clientEvent, error) {
	if update == nil {
		return nil, fmt.Errorf("%w: empty session update", provider.ErrInvalidRealtimeConfig)
	}
	session := map[string]any{"type": "realtime"}
	if update.System != nil {
		session["instructions"] = *update.System
	}
	if update.Tools != nil {
		tools, err := BuildTools(update.Tools)
		if err != nil {
			return nil, err
		}
		session["tools"] = tools
	}
	merged, err := mergeProviderOptions(session, update.ProviderOptions)
	if err != nil {
		return nil, err
	}
	if len(merged) == 1 {
		return nil, fmt.Errorf("%w: empty session update", provider.ErrInvalidRealtimeConfig)
	}
	return clientEvent{"type": "session.update", "session": merged}, nil
}
