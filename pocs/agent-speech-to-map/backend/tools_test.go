package main

import (
	"context"
	"strings"
	"testing"
)

func TestDispatchUnknownToolReportsError(t *testing.T) {
	res := dispatchTool(context.Background(), "no_such_tool", "{}")
	if !strings.Contains(res.content, "unknown tool") {
		t.Fatalf("an unknown tool name must surface as an error the model can recover from; got %q", res.content)
	}
	if res.places != nil || res.route != nil {
		t.Fatalf("a failed tool call must not contribute places or a route to the answer")
	}
}

func TestToolErrorIsValidJSON(t *testing.T) {
	got := toolError(context.DeadlineExceeded)
	if !strings.HasPrefix(got, `{"error":`) {
		t.Fatalf("tool errors must be JSON so the model receives a parseable tool message; got %q", got)
	}
}
