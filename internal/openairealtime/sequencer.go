package openairealtime

import (
	"strconv"
	"strings"
	"sync"
)

// sequencer orders response.create like the official Agents SDK: never while a
// response is active or another create is unconfirmed. Automatic requests in a
// row coalesce into one; requests with parameters are sent one by one, in order.
type sequencer struct {
	mu       sync.Mutex
	queue    []createRequest
	inFlight string // event_id of a sent response.create not yet confirmed
	ongoing  bool
	activeID string
	counter  int

	// The tool loop of the conversation: once every call of a finished
	// response has its output, one automatic response continues it.
	calls     map[string]bool
	toolTurn  bool
	respDone  bool
	outOfBand map[string]bool // responses outside the default conversation
}

// createRequest is a queued response.create; nil params is an automatic one.
type createRequest struct {
	params map[string]any
}

func newSequencer() *sequencer {
	return &sequencer{calls: map[string]bool{}, outOfBand: map[string]bool{}}
}

// request queues a response; an automatic request right after another
// automatic one is already covered by it.
func (s *sequencer) request(params map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.enqueue(params)
}

func (s *sequencer) enqueue(params map[string]any) {
	if params == nil && len(s.queue) > 0 && s.queue[len(s.queue)-1].params == nil {
		return
	}
	s.queue = append(s.queue, createRequest{params: params})
}

// next returns the response.create to send now, if the slot is free.
func (s *sequencer) next() (clientEvent, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ongoing || s.inFlight != "" || len(s.queue) == 0 {
		return nil, false
	}
	req := s.queue[0]
	s.queue = s.queue[1:]
	s.counter++
	s.inFlight = "goai_response_create_" + strconv.Itoa(s.counter)
	ev := clientEvent{"type": "response.create", "event_id": s.inFlight}
	if req.params != nil {
		ev["response"] = req.params
	}
	return ev, true
}

// sendFailed frees the slot of a response.create that could not be written.
func (s *sequencer) sendFailed(eventID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.inFlight == eventID {
		s.inFlight = ""
	}
}

// created marks a response active. A response in the default conversation
// starts a new tool turn.
func (s *sequencer) created(responseID string, inConversation bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ongoing, s.activeID, s.inFlight = true, responseID, ""
	if !inConversation {
		s.outOfBand[responseID] = true
		return
	}
	s.toolTurn, s.respDone = false, false
	clear(s.calls)
}

// done ends the active response; the end of another response leaves it
// active, as in the official SDK.
func (s *sequencer) done(responseID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if responseID == "" || s.activeID == "" || responseID == s.activeID {
		s.ongoing, s.activeID, s.inFlight = false, "", ""
	}
	if s.outOfBand[responseID] {
		delete(s.outOfBand, responseID)
		return
	}
	s.respDone = true
	s.continueToolTurn()
}

// rejected drops the unconfirmed response.create a server error refers to,
// instead of retrying it: a response the server already started covers it.
func (s *sequencer) rejected(e *wireError) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e == nil || s.inFlight == "" || s.ongoing {
		return false
	}
	matches := e.EventID == s.inFlight ||
		e.EventID == "" && (e.Code == "bad_response_create" || strings.Contains(strings.ToLower(e.Message), "response.create"))
	if matches {
		s.inFlight = ""
	}
	return matches
}

// toolCall records a function call of a response in the default conversation.
func (s *sequencer) toolCall(callID, responseID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.outOfBand[responseID] {
		return
	}
	s.calls[callID] = true
	s.toolTurn = true
}

// toolOutput records an output; the last one of a finished response queues
// the response that continues the conversation.
func (s *sequencer) toolOutput(callID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.calls, callID)
	s.continueToolTurn()
}

func (s *sequencer) continueToolTurn() {
	if s.toolTurn && s.respDone && len(s.calls) == 0 {
		s.toolTurn = false
		s.enqueue(nil)
	}
}

func (s *sequencer) active() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ongoing
}
