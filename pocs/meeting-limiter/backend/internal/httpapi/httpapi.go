package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"meetinglimiter/internal/calendar"
	"meetinglimiter/internal/logx"
	"meetinglimiter/internal/people"
)

type calendarService interface {
	Meetings(ctx context.Context, from, to time.Time) ([]calendar.Meeting, error)
	Wallets(ctx context.Context, at time.Time) ([]calendar.Wallet, error)
	Quote(ctx context.Context, r calendar.Request) (calendar.Quote, error)
	Book(ctx context.Context, r calendar.Request) (calendar.Meeting, error)
	Cancel(ctx context.Context, id string) error
	Rankings(ctx context.Context, from, to time.Time, limit int) (calendar.Rankings, error)
}

type userLister interface {
	All() []people.User
}

type Handler struct {
	calendar calendarService
	users    userLister
	logger   *slog.Logger
}

type policyView struct {
	people.Policy
	WeeklyBudget int64 `json:"weeklyBudget"`
}

type errorBody struct {
	Error string `json:"error"`
}

type shortBody struct {
	Error string         `json:"error"`
	Quote calendar.Quote `json:"quote"`
}

func New(c calendarService, u userLister, logger *slog.Logger) http.Handler {
	h := &Handler{calendar: c, users: u, logger: logger}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", h.health)
	mux.HandleFunc("GET /api/policies", h.policies)
	mux.HandleFunc("GET /api/users", h.listUsers)
	mux.HandleFunc("GET /api/meetings", h.listMeetings)
	mux.HandleFunc("POST /api/meetings/quote", h.quote)
	mux.HandleFunc("POST /api/meetings", h.book)
	mux.HandleFunc("DELETE /api/meetings/{id}", h.cancel)
	mux.HandleFunc("GET /api/wallets", h.wallets)
	mux.HandleFunc("GET /api/rankings", h.rankings)
	return logx.Middleware(mux)
}

func (h *Handler) health(w http.ResponseWriter, r *http.Request) {
	h.write(r.Context(), w, http.StatusOK, map[string]string{"status": "ok"})
}

func (h *Handler) policies(w http.ResponseWriter, r *http.Request) {
	var out []policyView
	for _, p := range people.Policies() {
		out = append(out, policyView{Policy: p, WeeklyBudget: int64(p.WeeklyBudget())})
	}
	h.write(r.Context(), w, http.StatusOK, out)
}

func (h *Handler) listUsers(w http.ResponseWriter, r *http.Request) {
	h.write(r.Context(), w, http.StatusOK, h.users.All())
}

func (h *Handler) listMeetings(w http.ResponseWriter, r *http.Request) {
	from, err := parseTime(r, "from")
	if err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	to, err := parseTime(r, "to")
	if err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	meetings, err := h.calendar.Meetings(r.Context(), from, to)
	if err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	h.write(r.Context(), w, http.StatusOK, nonNil(meetings))
}

func (h *Handler) quote(w http.ResponseWriter, r *http.Request) {
	req, err := decodeRequest(r)
	if err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	q, err := h.calendar.Quote(r.Context(), req)
	if err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	h.write(r.Context(), w, http.StatusOK, q)
}

func (h *Handler) book(w http.ResponseWriter, r *http.Request) {
	req, err := decodeRequest(r)
	if err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	m, err := h.calendar.Book(r.Context(), req)
	if err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	h.logger.InfoContext(r.Context(), "meeting booked", "meeting_id", m.ID, "total_cost", m.TotalCost.String())
	h.write(r.Context(), w, http.StatusCreated, m)
}

func (h *Handler) cancel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := h.calendar.Cancel(r.Context(), id); err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	h.logger.InfoContext(r.Context(), "meeting cancelled", "meeting_id", id)
	w.WriteHeader(http.StatusNoContent)
}

func (h *Handler) wallets(w http.ResponseWriter, r *http.Request) {
	at, err := parseTime(r, "week")
	if err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	wallets, err := h.calendar.Wallets(r.Context(), at)
	if err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	h.write(r.Context(), w, http.StatusOK, wallets)
}

func (h *Handler) rankings(w http.ResponseWriter, r *http.Request) {
	from, err := parseTime(r, "from")
	if err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	to, err := parseTime(r, "to")
	if err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	limit, err := strconv.Atoi(r.URL.Query().Get("limit"))
	if err != nil || limit < 1 || limit > 50 {
		h.fail(r.Context(), w, fmt.Errorf("%w: limit must be 1 to 50", calendar.ErrInvalid))
		return
	}
	out, err := h.calendar.Rankings(r.Context(), from, to, limit)
	if err != nil {
		h.fail(r.Context(), w, err)
		return
	}
	h.write(r.Context(), w, http.StatusOK, out)
}

func (h *Handler) fail(ctx context.Context, w http.ResponseWriter, err error) {
	var short *calendar.InsufficientFundsError
	switch {
	case errors.As(err, &short):
		h.write(ctx, w, http.StatusPaymentRequired, shortBody{Error: err.Error(), Quote: short.Quote})
	case errors.Is(err, calendar.ErrInvalid):
		h.write(ctx, w, http.StatusBadRequest, errorBody{Error: err.Error()})
	case errors.Is(err, calendar.ErrNotFound):
		h.write(ctx, w, http.StatusNotFound, errorBody{Error: err.Error()})
	default:
		h.logger.ErrorContext(ctx, "request failed", "err", err)
		h.write(ctx, w, http.StatusInternalServerError, errorBody{Error: "internal error"})
	}
}

func (h *Handler) write(ctx context.Context, w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(body); err != nil {
		h.logger.ErrorContext(ctx, "encode response", "err", err)
	}
}

func decodeRequest(r *http.Request) (calendar.Request, error) {
	var req calendar.Request
	dec := json.NewDecoder(http.MaxBytesReader(nil, r.Body, 1<<16))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		return calendar.Request{}, fmt.Errorf("%w: %w", calendar.ErrInvalid, err)
	}
	return req, nil
}

func parseTime(r *http.Request, name string) (time.Time, error) {
	t, err := time.Parse(time.RFC3339, r.URL.Query().Get(name))
	if err != nil {
		return time.Time{}, fmt.Errorf("%w: query %s must be RFC3339", calendar.ErrInvalid, name)
	}
	return t, nil
}

func nonNil[T any](s []T) []T {
	if s == nil {
		return []T{}
	}
	return s
}
