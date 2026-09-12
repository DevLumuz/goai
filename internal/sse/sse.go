// Package sse provides a scanner for Server-Sent Events (SSE) streams.
//
// It handles data payloads, complete event framing, and the [DONE] sentinel.
// JSON deserialization is left to the caller.
package sse

import (
	"bytes"
	"errors"
	"fmt"
	"io"
)

// MaxLineSize is the upper bound on a single SSE line, in bytes.
//
// Large enough to comfortably hold long tool-call argument deltas and
// reasoning blocks observed from production providers, while preventing a
// malicious or buggy upstream from forcing unbounded client-side allocation
// (a DoS vector). Lines exceeding this size cause Next to stop and Err to
// report the violation.
const MaxLineSize = 16 << 20 // 16 MiB

// MaxEventSize is the upper bound on the aggregate data payload of one SSE
// event, in bytes. Multi-line data fields are joined with a single newline.
const MaxEventSize = 16 << 20 // 16 MiB

// Event is one complete SSE event. Type is empty when the event field was not
// present. Data contains all data fields joined with a newline.
type Event struct {
	Type string
	Data []byte
}

// Scanner reads SSE data payloads from an io.Reader.
type Scanner struct {
	reader   io.Reader
	readBuf  [4096]byte
	pending  bytes.Buffer
	scanFrom int
	readErr  error
	atStart  bool
	skipLF   bool
	err      error
	done     bool
}

// NewScanner creates an SSE scanner for r.
//
// Lines up to [MaxLineSize] bytes are accepted; longer lines cause the
// scanner to stop with an error reported via Err. This lifts the 1 MiB
// limit imposed by bufio.Scanner while still bounding memory use.
func NewScanner(r io.Reader) *Scanner {
	return &Scanner{reader: r, atStart: true}
}

// NextLine returns the next line from the SSE stream with trailing CR/LF
// stripped. Unlike [Scanner.Next], it returns every line including blank
// lines and non-"data:" lines, so callers parsing event-typed SSE
// (interleaved "event:" + "data:" pairs, e.g. the OpenAI Responses API)
// can implement their own line dispatch while still benefiting from
// [MaxLineSize]-bounded reads. Returns ("", false) at EOF or after a read
// error (which is reported via [Scanner.Err]).
//
// Mix Next and NextLine on the same scanner at your own risk; pick one
// mode per stream.
func (s *Scanner) NextLine() (line string, ok bool) {
	if s.err != nil {
		return "", false
	}
	raw, err := s.readLine()
	if err != nil {
		if !errors.Is(err, io.EOF) {
			s.err = err
		}
		if len(raw) == 0 {
			return "", false
		}
	}
	return string(bytes.TrimRight(raw, "\r\n")), true
}

// NextEvent returns the next complete SSE event. Comments and events without
// data fields are ignored. An event is complete only after a blank line; any
// pending event is discarded at EOF. Field values may omit the optional space
// after the colon.
//
// Mix NextEvent with Next or NextLine on the same scanner at your own risk;
// pick one mode per stream.
func (s *Scanner) NextEvent() (Event, bool) {
	if s.err != nil {
		return Event{}, false
	}

	var event Event
	var hasData bool
	for {
		raw, err := s.readLine()
		if len(raw) > 0 {
			line := bytes.TrimRight(raw, "\r\n")
			if len(line) == 0 {
				if hasData {
					return event, true
				}
				event.Type = ""
			} else if line[0] != ':' {
				field, value, _ := bytes.Cut(line, []byte(":"))
				value = bytes.TrimPrefix(value, []byte(" "))
				switch string(field) {
				case "event":
					event.Type = string(value)
				case "data":
					additional := len(value)
					if hasData {
						additional++
					}
					if len(event.Data)+additional > MaxEventSize {
						s.err = fmt.Errorf("sse: event exceeds %d bytes", MaxEventSize)
						return Event{}, false
					}
					if hasData {
						event.Data = append(event.Data, '\n')
					}
					event.Data = append(event.Data, value...)
					hasData = true
				}
			}
		}

		if err != nil {
			if !errors.Is(err, io.EOF) {
				s.err = err
			}
			return Event{}, false
		}
	}
}

// Next returns the next SSE data payload (with "data: " prefix stripped).
// Returns ("", false) at EOF or after [DONE].
// Skips non-"data: " lines and blank lines.
func (s *Scanner) Next() (data string, ok bool) {
	if s.done || s.err != nil {
		return "", false
	}

	for {
		line, err := s.readLine()
		if len(line) > 0 {
			line = bytes.TrimRight(line, "\r\n")
			if after, ok0 := bytes.CutPrefix(line, []byte("data:")); ok0 {
				payload := bytes.TrimPrefix(after, []byte(" "))
				if bytes.Equal(payload, []byte("[DONE]")) {
					s.done = true
					return "", false
				}
				return string(payload), true
			}
			// Non-"data:" line: skip and continue reading.
		}

		if err != nil {
			if !errors.Is(err, io.EOF) {
				s.err = err
			}
			return "", false
		}
	}
}

// readLine reads one CR, LF, or CRLF-terminated line with fixed-size reads. It
// enforces [MaxLineSize] to prevent unbounded memory growth from a hostile or
// malformed stream.
//
// Returns the line (including its terminator, when present) together with any
// terminal error. A final partial line at EOF is returned with io.EOF. One
// leading UTF-8 BOM is stripped from the stream before field parsing.
// The returned bytes are valid only until the next readLine call. Public
// scanner methods copy any data they return to callers.
func (s *Scanner) readLine() ([]byte, error) {
	const maxConsecutiveEmptyReads = 100
	emptyReads := 0

	for {
		if s.skipLF {
			if s.pending.Len() > 0 {
				if s.pending.Bytes()[0] == '\n' {
					s.pending.Next(1)
				}
				s.skipLF = false
			} else if s.readErr != nil {
				s.skipLF = false
			}
		}

		if end, complete := s.lineEnd(); complete {
			if end > MaxLineSize {
				return nil, fmt.Errorf("sse: line exceeds %d bytes", MaxLineSize)
			}
			line := s.pending.Next(end)
			s.scanFrom = 0
			// Reuse ordinary read buffers, but release oversized drained lines.
			if s.pending.Len() == 0 && s.pending.Cap() > len(s.readBuf) {
				s.pending = bytes.Buffer{}
			}
			return s.stripInitialBOM(line), nil
		}

		if s.pending.Len() > MaxLineSize {
			return nil, fmt.Errorf("sse: line exceeds %d bytes", MaxLineSize)
		}
		if s.readErr != nil {
			if s.pending.Len() == 0 {
				return nil, s.readErr
			}
			line := s.pending.Bytes()
			s.pending = bytes.Buffer{}
			s.scanFrom = 0
			return s.stripInitialBOM(line), s.readErr
		}

		n, err := s.reader.Read(s.readBuf[:])
		if n > 0 {
			_, _ = s.pending.Write(s.readBuf[:n])
			emptyReads = 0
		} else if err == nil {
			emptyReads++
			if emptyReads >= maxConsecutiveEmptyReads {
				s.readErr = io.ErrNoProgress
			}
		}
		if err != nil {
			s.readErr = err
		}
	}
}

func (s *Scanner) lineEnd() (int, bool) {
	index := bytes.IndexAny(s.pending.Bytes()[s.scanFrom:], "\r\n")
	if index < 0 {
		s.scanFrom = s.pending.Len()
		return 0, false
	}
	index += s.scanFrom

	end := index + 1
	if s.pending.Bytes()[index] == '\r' {
		s.skipLF = true
	}
	return end, true
}

func (s *Scanner) stripInitialBOM(line []byte) []byte {
	if !s.atStart {
		return line
	}
	s.atStart = false
	return bytes.TrimPrefix(line, []byte{0xef, 0xbb, 0xbf})
}

// Err returns the first non-EOF error encountered while reading,
// including line-size violations.
func (s *Scanner) Err() error {
	return s.err
}

// IsDone reports whether the scanner has encountered the [DONE] sentinel.
func (s *Scanner) IsDone() bool {
	return s.done
}
