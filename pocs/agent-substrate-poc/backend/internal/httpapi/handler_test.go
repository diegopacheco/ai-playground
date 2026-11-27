package httpapi

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"agentsubstratepoc/internal/agent"
	"agentsubstratepoc/internal/ate"
	"agentsubstratepoc/internal/console"
	"agentsubstratepoc/internal/logx"
	"agentsubstratepoc/internal/proxy"
)

type stubService struct {
	err     error
	created string
}

func (s *stubService) Actors(context.Context) ([]ate.Actor, error) {
	return []ate.Actor{{Name: "alice", State: ate.StateSuspended}}, s.err
}
func (s *stubService) Workers(context.Context) ([]ate.Worker, error) { return nil, s.err }
func (s *stubService) Actor(_ context.Context, n string) (ate.Actor, error) {
	return ate.Actor{Name: n}, s.err
}
func (s *stubService) Create(_ context.Context, n string) (ate.Actor, error) {
	s.created = n
	return ate.Actor{Name: n}, s.err
}
func (s *stubService) Suspend(_ context.Context, n string) (ate.Actor, error) {
	return ate.Actor{Name: n}, s.err
}
func (s *stubService) Resume(_ context.Context, n string) (ate.Actor, error) {
	return ate.Actor{Name: n}, s.err
}
func (s *stubService) Delete(context.Context, string) error { return s.err }
func (s *stubService) State(context.Context, string) (agent.State, error) {
	return agent.State{BootID: "b"}, s.err
}
func (s *stubService) Remember(_ context.Context, _ string, text string) (agent.Note, error) {
	return agent.Note{ID: 1, Text: text}, s.err
}

func do(t *testing.T, svc *stubService, method, path, body string) (*http.Response, string) {
	t.Helper()
	h := New(svc, slog.New(slog.NewTextHandler(io.Discard, nil)))
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec.Result(), rec.Body.String()
}

func TestCreateReadsTheNameFromJSON(t *testing.T) {
	svc := &stubService{}
	res, _ := do(t, svc, "POST", "/api/actors", `{"name":"alice"}`)
	if res.StatusCode != http.StatusCreated || svc.created != "alice" {
		t.Errorf("status=%d created=%q", res.StatusCode, svc.created)
	}
}

func TestBadJSONIs400(t *testing.T) {
	res, _ := do(t, &stubService{}, "POST", "/api/actors/alice/notes", "{")
	if res.StatusCode != http.StatusBadRequest {
		t.Errorf("status=%d", res.StatusCode)
	}
}

func TestDomainErrorsMapToHTTPStatus(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want int
	}{
		{"bad actor name is the client's fault", fmt.Errorf("%w: x", console.ErrInvalid), http.StatusBadRequest},
		{"unknown actor", fmt.Errorf("get: %w", ate.ErrNotFound), http.StatusNotFound},
		{"duplicate actor", ate.ErrAlreadyExists, http.StatusConflict},
		{"suspending a suspended actor", ate.ErrConflict, http.StatusConflict},
		{"too many open proxies", proxy.ErrNoFreePort, http.StatusServiceUnavailable},
		{"cluster is down", errors.New("dial tcp: refused"), http.StatusBadGateway},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, body := do(t, &stubService{err: tt.err}, "POST", "/api/actors/alice/suspend", "")
			if res.StatusCode != tt.want {
				t.Errorf("status = %d, want %d", res.StatusCode, tt.want)
			}
			if !strings.Contains(body, "error") {
				t.Errorf("body must carry the error: %s", body)
			}
		})
	}
}

func TestCorrelationIDIsReturned(t *testing.T) {
	h := New(&stubService{}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	req := httptest.NewRequest("GET", "/api/actors", nil)
	req.Header.Set(logx.Header, "abc")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Header().Get(logx.Header) != "abc" {
		t.Errorf("header = %q", rec.Header().Get(logx.Header))
	}
}
