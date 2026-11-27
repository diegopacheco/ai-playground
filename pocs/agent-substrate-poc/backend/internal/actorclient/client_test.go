package actorclient

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"agentsubstratepoc/internal/agent"
	"agentsubstratepoc/internal/logx"
)

func serve(t *testing.T, h http.HandlerFunc) int {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	_, p, _ := net.SplitHostPort(srv.Listener.Addr().String())
	port, _ := strconv.Atoi(p)
	return port
}

func TestCorrelationIDIsForwardedToTheActor(t *testing.T) {
	var seen string
	port := serve(t, func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get(logx.Header)
		_ = json.NewEncoder(w).Encode(agent.State{BootID: "b1"})
	})
	ctx := logx.With(context.Background(), logx.Fields{CorrelationID: "cid-1"})
	s, err := New("127.0.0.1", time.Second).State(ctx, port)
	if err != nil {
		t.Fatal(err)
	}
	if seen != "cid-1" || s.BootID != "b1" {
		t.Errorf("seen=%q state=%+v", seen, s)
	}
}

func TestUnexpectedStatusIsAnError(t *testing.T) {
	port := serve(t, func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "note text is empty", http.StatusBadRequest)
	})
	_, err := New("127.0.0.1", time.Second).AddNote(context.Background(), port, " ")
	if err == nil {
		t.Fatal("want error for 400")
	}
}
