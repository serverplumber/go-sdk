// Copyright 2026 The Go MCP SDK Authors. All rights reserved.
// Use of this source code is governed by the license
// that can be found in the LICENSE file.

package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/modelcontextprotocol/go-sdk/internal/jsonrpc2"
	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
)

// Tests for go-sdk#1209: a bad inbound frame on the stdio transport must be
// answered with an error and must not end the session.

const (
	stdioInit    = `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"v0"}}}`
	stdioInited  = `{"jsonrpc":"2.0","method":"notifications/initialized"}`
	stdioPing2   = `{"jsonrpc":"2.0","id":2,"method":"ping"}`
	stdioPing3   = `{"jsonrpc":"2.0","id":3,"method":"ping"}`
	ioReadExpiry = time.Second
)

// badFrames are the frames from go-sdk#1209 that must be answered with an
// error and must not end the session.
var badFrames = []struct {
	name     string
	frame    string
	wantCode int64
	// wantID is the id the error response carries, as replySummary renders it.
	// MCP's JSONRPCErrorResponse has an optional id that is a string or a
	// number, never null, so an id that cannot be echoed is "absent".
	wantID string
	// sessionOnly marks a frame the transport decodes as a valid message, so
	// only the session can reject it.
	sessionOnly bool
}{
	{name: "syntax", frame: `{bad json`, wantCode: -32700, wantID: "absent"},
	// The decoder only fails on the next line, so resynchronising from the
	// point of failure would swallow the frame after this one.
	{name: "unbalanced", frame: `{"jsonrpc":"2.0","id":9,"method":"ping"`, wantCode: -32700, wantID: "absent"},
	{name: "nul byte", frame: "{\"jsonrpc\":\"2.0\",\x00\"id\":9,\"method\":\"ping\"}", wantCode: -32700, wantID: "absent"},
	{name: "deep nesting", frame: strings.Repeat("[", 11000), wantCode: -32700, wantID: "absent"},
	{name: "number", frame: `42`, wantCode: -32600, wantID: "absent"},
	{name: "empty object", frame: `{}`, wantCode: -32600, wantID: "absent"},
	{name: "empty batch", frame: `[]`, wantCode: -32600, wantID: "absent"},
	// MakeID reports these as ErrParse, but the JSON is valid, so they are
	// invalid requests, not parse errors.
	{name: "boolean id", frame: `{"jsonrpc":"2.0","id":true,"method":"ping"}`, wantCode: -32600, wantID: "absent"},
	{name: "object id", frame: `{"jsonrpc":"2.0","id":{},"method":"ping"}`, wantCode: -32600, wantID: "absent"},
	{name: "array id", frame: `{"jsonrpc":"2.0","id":[],"method":"ping"}`, wantCode: -32600, wantID: "absent"},
	// A frame without a method is shaped like a response, so its id names one
	// of the server's own requests and echoing it would mislead the client.
	{name: "error not an object", frame: `{"jsonrpc":"2.0","id":9,"error":"x"}`, wantCode: -32600, wantID: "absent"},
	{name: "id, no method", frame: `{"jsonrpc":"2.0","id":9}`, wantCode: -32600, wantID: "absent", sessionOnly: true},
}

// runStdioSession serves s over an [IOTransport], writes input to it in a
// single write, and returns every line the server writes. Stdin is closed once
// n lines have arrived, or after a timeout.
//
// Delivering input all at once is what makes the replies already owed race the
// read error, so callers should not split it up.
func runStdioSession(t *testing.T, s *Server, input string, n int) []json.RawMessage {
	t.Helper()
	// Pipes rather than a strings.Reader and a bytes.Buffer: a strings.Reader
	// reaches EOF straight after the input, ending the session before the
	// replies under test are written, and reading a bytes.Buffer while the
	// server writes to it is a data race.
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	errc := make(chan error, 1)
	go func() { errc <- s.Run(ctx, &IOTransport{Reader: inR, Writer: outW}) }()

	go inW.Write([]byte(input))

	lines := make(chan json.RawMessage)
	var scanErr error
	go func() {
		defer close(lines)
		sc := bufio.NewScanner(outR)
		for sc.Scan() {
			lines <- json.RawMessage(bytes.Clone(sc.Bytes()))
		}
		scanErr = sc.Err()
	}()

	var got []json.RawMessage
wait:
	for len(got) < n {
		select {
		case l, ok := <-lines:
			if !ok {
				break wait
			}
			got = append(got, l)
		case <-ctx.Done():
			break wait
		}
	}

	inW.Close()
	<-errc
	// Closing stdout ends the scan, so lines written after the first n, such
	// as a duplicate error reply, are collected too.
	outW.Close()
	for l := range lines {
		got = append(got, l)
	}
	if scanErr != nil {
		t.Errorf("reading server output: %v", scanErr)
	}
	return got
}

// replySummary reduces each reply in lines to "<id> result" or
// "<id> error <code>", sorted, so that replies arriving in any order compare
// equal. A reply with no id member is rendered as "absent", and one with
// "id": null as "null". Server-to-client requests and notifications are
// dropped.
func replySummary(t *testing.T, lines []json.RawMessage) []string {
	t.Helper()
	var got []string
	for _, l := range lines {
		var r struct {
			ID     json.RawMessage       `json:"id"`
			Method string                `json:"method"`
			Error  *struct{ Code int64 } `json:"error"`
		}
		if err := json.Unmarshal(l, &r); err != nil {
			t.Fatalf("server wrote a line that is not a JSON-RPC message: %s: %v", l, err)
		}
		if r.Method != "" {
			continue
		}
		id := string(r.ID)
		if id == "" {
			id = "absent"
		}
		if r.Error != nil {
			got = append(got, fmt.Sprintf("%s error %d", id, r.Error.Code))
		} else {
			got = append(got, fmt.Sprintf("%s result", id))
		}
	}
	slices.Sort(got)
	return got
}

// wantReplies is what a session owes for initialize, ping 2, a bad frame and
// ping 3, in the order replySummary produces.
func wantReplies(id string, code int64) []string {
	want := []string{"1 result", "2 result", "3 result", fmt.Sprintf("%s error %d", id, code)}
	slices.Sort(want)
	return want
}

// TestStdioBadFrameKeepsSession sends initialize, notifications/initialized,
// ping 2, a bad frame and ping 3 in one write. The session must answer
// initialize and both pings, and answer the bad frame with exactly one error.
//
// The failure depends on timing, so run it with -count and -race.
func TestStdioBadFrameKeepsSession(t *testing.T) {
	for _, tt := range badFrames {
		t.Run(tt.name, func(t *testing.T) {
			s := NewServer(&Implementation{Name: "test", Version: "v0"}, nil)
			input := strings.Join([]string{stdioInit, stdioInited, stdioPing2, tt.frame, stdioPing3}, "\n") + "\n"

			got := replySummary(t, runStdioSession(t, s, input, 4))
			if diff := cmp.Diff(wantReplies(tt.wantID, tt.wantCode), got); diff != "" {
				t.Errorf("replies mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestStdioBadFrameKeepsInFlightReplies holds the ping 2 handler until ping 3
// has been received, so the bad frame between them is read while ping 2 is
// still in flight. Pong 2 must still be written.
func TestStdioBadFrameKeepsInFlightReplies(t *testing.T) {
	for _, tt := range badFrames {
		t.Run(tt.name, func(t *testing.T) {
			s := NewServer(&Implementation{Name: "test", Version: "v0"}, nil)
			ping3Seen := make(chan struct{})
			markPing3Seen := sync.OnceFunc(func() { close(ping3Seen) })
			s.AddReceivingMiddleware(func(next MethodHandler) MethodHandler {
				return func(ctx context.Context, method string, req Request) (Result, error) {
					if method == methodPing {
						switch ctx.Value(idContextKey{}) {
						case jsonrpc2.Int64ID(2):
							// Bounded, because a session that has already ended
							// never delivers ping 3, and closing it waits for
							// in-flight handlers.
							select {
							case <-ping3Seen:
							case <-time.After(2 * time.Second):
							}
						case jsonrpc2.Int64ID(3):
							markPing3Seen()
						}
					}
					return next(ctx, method, req)
				}
			})
			input := strings.Join([]string{stdioInit, stdioInited, stdioPing2, tt.frame, stdioPing3}, "\n") + "\n"

			got := replySummary(t, runStdioSession(t, s, input, 4))
			if diff := cmp.Diff(wantReplies(tt.wantID, tt.wantCode), got); diff != "" {
				t.Errorf("replies mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestStdioBlankLineGetsNoReply sends a blank line between ping 2 and ping 3.
// A blank line is not a frame, so unlike a bad frame it must get no reply at
// all, and the session must carry on. The "\r" case is a blank line in a
// stream with Windows line endings.
func TestStdioBlankLineGetsNoReply(t *testing.T) {
	for _, blank := range []string{"", "\r"} {
		t.Run(fmt.Sprintf("%q", blank), func(t *testing.T) {
			s := NewServer(&Implementation{Name: "test", Version: "v0"}, nil)
			input := strings.Join([]string{stdioInit, stdioInited, stdioPing2, blank, stdioPing3}, "\n") + "\n"

			got := replySummary(t, runStdioSession(t, s, input, 3))
			want := []string{"1 result", "2 result", "3 result"}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("replies mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// readOrTimeout calls tr.Read, failing the test if it blocks for longer than
// ioReadExpiry. A transport that stopped reading after an error blocks forever.
func readOrTimeout(t *testing.T, tr *ioConn) (jsonrpc.Message, error) {
	t.Helper()
	type result struct {
		msg jsonrpc.Message
		err error
	}
	done := make(chan result, 1)
	go func() {
		msg, err := tr.Read(context.Background())
		done <- result{msg, err}
	}()
	select {
	case r := <-done:
		return r.msg, r.err
	case <-time.After(ioReadExpiry):
		t.Fatal("Read() blocked: the transport stopped reading")
		return nil, nil
	}
}

// readUntilEOF reads from tr until io.EOF, skipping the errors a transport may
// report for a bad frame, and returns the messages read. It allows a bounded
// number of Reads, so a transport that keeps failing does not loop forever.
func readUntilEOF(t *testing.T, tr *ioConn) []jsonrpc.Message {
	t.Helper()
	var msgs []jsonrpc.Message
	for range 4 {
		msg, err := readOrTimeout(t, tr)
		if errors.Is(err, io.EOF) {
			return msgs
		}
		if err == nil {
			msgs = append(msgs, msg)
		}
	}
	t.Fatalf("Read() did not reach io.EOF; read %v", msgs)
	return nil
}

// TestIOConnReadAfterBadFrame checks resynchronisation at the transport level:
// after a bad frame, the transport must deliver the next frame, once, and
// nothing for the bad frame itself.
//
// Whether Read reports the bad frame as an error first is a design choice, so
// the test accepts either.
func TestIOConnReadAfterBadFrame(t *testing.T) {
	for _, tt := range badFrames {
		if tt.sessionOnly {
			continue
		}
		t.Run(tt.name, func(t *testing.T) {
			tr := newIOConn(rwc{rc: io.NopCloser(strings.NewReader(tt.frame + "\n" + stdioPing3 + "\n"))})
			t.Cleanup(func() { tr.Close() })

			msgs := readUntilEOF(t, tr)
			if len(msgs) != 1 {
				t.Fatalf("read %d messages %v, want only ping 3", len(msgs), msgs)
			}
			req, ok := msgs[0].(*jsonrpc.Request)
			if !ok || req.Method != methodPing || req.ID != jsonrpc2.Int64ID(3) {
				t.Errorf("read %v, want ping 3", msgs[0])
			}
		})
	}
}

// TestIOConnBadFrameAtEOF covers a bad frame that is the last thing in the
// stream, with and without a trailing newline. Read must reach io.EOF without
// delivering a message. Resynchronising to the next newline must not block
// when there is none.
func TestIOConnBadFrameAtEOF(t *testing.T) {
	for _, tt := range badFrames {
		if tt.sessionOnly {
			continue
		}
		for _, end := range []string{"", "\n"} {
			t.Run(fmt.Sprintf("%s/%q", tt.name, end), func(t *testing.T) {
				tr := newIOConn(rwc{rc: io.NopCloser(strings.NewReader(tt.frame + end))})
				t.Cleanup(func() { tr.Close() })

				if msgs := readUntilEOF(t, tr); len(msgs) != 0 {
					t.Errorf("read %v, want no messages", msgs)
				}
			})
		}
	}
}

// TestStdioInFlightRepliesAtEOF holds the ping 2 handler in receiving
// middleware until the reader has hit EOF, then releases it.
//
// Undecided: whether pong 2 must still be written. Stdin closing does not mean
// stdout has gone. Pin whichever behaviour is chosen.
func TestStdioInFlightRepliesAtEOF(t *testing.T) {
	t.Skip("blocked on deciding whether replies owed at EOF must be written")
}

// TestIOConnOversizedFrame sends a frame larger than the newIOConnLimited cap,
// then ping 3. An oversized frame is a bad frame like any other: the transport
// must discard the rest of it without buffering it, reset the cap, and deliver
// ping 3.
//
// Whether Read reports errFrameTooLarge first is a design choice, so the test
// accepts either. [TestIOConnFrameCap] checks that the cap trips at all,
// including on an endless frame with no newline.
//
// Framing on lines changes one [TestIOConnFrameCap] case: "cap does not reset
// per new line" assumes a value can span lines, which the stdio transport
// forbids ("MUST NOT contain embedded newlines").
func TestIOConnOversizedFrame(t *testing.T) {
	const limit = 1024
	tests := []struct {
		name  string
		frame string
	}{
		// A valid request, so a transport that buffered past the cap would
		// deliver it.
		{"request", `{"jsonrpc":"2.0","id":9,"method":"ping","params":{"pad":"` + strings.Repeat("A", 2*limit) + `"}}`},
		{"garbage", strings.Repeat("A", 2*limit)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// ping 3 is far under the cap, so it is only delivered if the cap
			// was reset after the oversized frame.
			tr := newIOConnLimited(rwc{rc: io.NopCloser(strings.NewReader(tt.frame + "\n" + stdioPing3 + "\n"))}, limit)
			t.Cleanup(func() { tr.Close() })

			msgs := readUntilEOF(t, tr)
			if len(msgs) != 1 {
				t.Fatalf("read %d messages %v, want only ping 3", len(msgs), msgs)
			}
			req, ok := msgs[0].(*jsonrpc.Request)
			if !ok || req.Method != methodPing || req.ID != jsonrpc2.Int64ID(3) {
				t.Errorf("read %v, want ping 3", msgs[0])
			}
		})
	}
}
