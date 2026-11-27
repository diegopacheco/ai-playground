package logx

import (
	"context"
	"crypto/rand"
	"log/slog"
	"net/http"
	"os"
)

type ctxKey struct{}

const Header = "X-Correlation-Id"

type Fields struct {
	CorrelationID string
	Path          string
	UserID        string
}

func With(ctx context.Context, f Fields) context.Context {
	return context.WithValue(ctx, ctxKey{}, f)
}

func CorrelationID(ctx context.Context) string {
	f, _ := ctx.Value(ctxKey{}).(Fields)
	return f.CorrelationID
}

type handler struct {
	slog.Handler
	service string
}

func (h handler) Handle(ctx context.Context, r slog.Record) error {
	f, _ := ctx.Value(ctxKey{}).(Fields)
	r.AddAttrs(
		slog.String("correlation_id", f.CorrelationID),
		slog.String("path", f.Path),
		slog.String("user_id", f.UserID),
		slog.String("service", h.service),
	)
	return h.Handler.Handle(ctx, r)
}

func (h handler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return handler{Handler: h.Handler.WithAttrs(attrs), service: h.service}
}

func (h handler) WithGroup(name string) slog.Handler {
	return handler{Handler: h.Handler.WithGroup(name), service: h.service}
}

func New(service string) *slog.Logger {
	return slog.New(handler{Handler: slog.NewJSONHandler(os.Stdout, nil), service: service})
}

func Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get(Header)
		if id == "" {
			id = rand.Text()
		}
		w.Header().Set(Header, id)
		ctx := With(r.Context(), Fields{CorrelationID: id, Path: r.URL.Path, UserID: "anonymous"})
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}
