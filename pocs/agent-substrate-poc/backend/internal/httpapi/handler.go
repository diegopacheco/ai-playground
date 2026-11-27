package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"

	"agentsubstratepoc/internal/agent"
	"agentsubstratepoc/internal/ate"
	"agentsubstratepoc/internal/console"
	"agentsubstratepoc/internal/logx"
	"agentsubstratepoc/internal/proxy"
)

type service interface {
	Actors(ctx context.Context) ([]ate.Actor, error)
	Workers(ctx context.Context) ([]ate.Worker, error)
	Actor(ctx context.Context, name string) (ate.Actor, error)
	Create(ctx context.Context, name string) (ate.Actor, error)
	Suspend(ctx context.Context, name string) (ate.Actor, error)
	Resume(ctx context.Context, name string) (ate.Actor, error)
	Delete(ctx context.Context, name string) error
	State(ctx context.Context, name string) (agent.State, error)
	Remember(ctx context.Context, name, text string) (agent.Note, error)
}

type Handler struct {
	svc    service
	logger *slog.Logger
}

func New(svc service, logger *slog.Logger) http.Handler {
	h := &Handler{svc: svc, logger: logger}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, r *http.Request) {
		h.write(w, r, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /api/actors", h.list)
	mux.HandleFunc("POST /api/actors", h.create)
	mux.HandleFunc("GET /api/actors/{name}", h.get)
	mux.HandleFunc("DELETE /api/actors/{name}", h.remove)
	mux.HandleFunc("POST /api/actors/{name}/suspend", h.suspend)
	mux.HandleFunc("POST /api/actors/{name}/resume", h.resume)
	mux.HandleFunc("GET /api/actors/{name}/state", h.state)
	mux.HandleFunc("POST /api/actors/{name}/notes", h.note)
	mux.HandleFunc("GET /api/workers", h.workers)
	return logx.Middleware(mux)
}

type nameRequest struct {
	Name string `json:"name"`
}

type noteRequest struct {
	Text string `json:"text"`
}

func (h *Handler) list(w http.ResponseWriter, r *http.Request) {
	actors, err := h.svc.Actors(r.Context())
	h.reply(w, r, http.StatusOK, actors, err)
}

func (h *Handler) workers(w http.ResponseWriter, r *http.Request) {
	workers, err := h.svc.Workers(r.Context())
	h.reply(w, r, http.StatusOK, workers, err)
}

func (h *Handler) get(w http.ResponseWriter, r *http.Request) {
	a, err := h.svc.Actor(r.Context(), r.PathValue("name"))
	h.reply(w, r, http.StatusOK, a, err)
}

func (h *Handler) create(w http.ResponseWriter, r *http.Request) {
	var req nameRequest
	if !h.decode(w, r, &req) {
		return
	}
	a, err := h.svc.Create(r.Context(), req.Name)
	h.reply(w, r, http.StatusCreated, a, err)
}

func (h *Handler) suspend(w http.ResponseWriter, r *http.Request) {
	a, err := h.svc.Suspend(r.Context(), r.PathValue("name"))
	h.reply(w, r, http.StatusOK, a, err)
}

func (h *Handler) resume(w http.ResponseWriter, r *http.Request) {
	a, err := h.svc.Resume(r.Context(), r.PathValue("name"))
	h.reply(w, r, http.StatusOK, a, err)
}

func (h *Handler) remove(w http.ResponseWriter, r *http.Request) {
	err := h.svc.Delete(r.Context(), r.PathValue("name"))
	h.reply(w, r, http.StatusOK, map[string]string{"deleted": r.PathValue("name")}, err)
}

func (h *Handler) state(w http.ResponseWriter, r *http.Request) {
	s, err := h.svc.State(r.Context(), r.PathValue("name"))
	h.reply(w, r, http.StatusOK, s, err)
}

func (h *Handler) note(w http.ResponseWriter, r *http.Request) {
	var req noteRequest
	if !h.decode(w, r, &req) {
		return
	}
	n, err := h.svc.Remember(r.Context(), r.PathValue("name"), req.Text)
	h.reply(w, r, http.StatusCreated, n, err)
}

func (h *Handler) decode(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096)).Decode(v); err != nil {
		h.write(w, r, http.StatusBadRequest, map[string]string{"error": "invalid json body"})
		return false
	}
	return true
}

func (h *Handler) reply(w http.ResponseWriter, r *http.Request, status int, v any, err error) {
	if err != nil {
		h.writeError(w, r, err)
		return
	}
	h.write(w, r, status, v)
}

func statusOf(err error) int {
	switch {
	case errors.Is(err, console.ErrInvalid):
		return http.StatusBadRequest
	case errors.Is(err, ate.ErrNotFound):
		return http.StatusNotFound
	case errors.Is(err, ate.ErrAlreadyExists), errors.Is(err, ate.ErrConflict):
		return http.StatusConflict
	case errors.Is(err, proxy.ErrNoFreePort):
		return http.StatusServiceUnavailable
	default:
		return http.StatusBadGateway
	}
}

func (h *Handler) writeError(w http.ResponseWriter, r *http.Request, err error) {
	status := statusOf(err)
	h.logger.WarnContext(r.Context(), "request failed", "status", status, "err", err)
	h.write(w, r, status, map[string]string{"error": err.Error()})
}

func (h *Handler) write(w http.ResponseWriter, r *http.Request, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		h.logger.ErrorContext(r.Context(), "encode response", "err", err)
	}
}
