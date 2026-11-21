package main

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"meetinglimiter/internal/calendar"
	"meetinglimiter/internal/config"
	"meetinglimiter/internal/httpapi"
	"meetinglimiter/internal/logx"
	"meetinglimiter/internal/people"
)

func main() {
	logger := logx.New("meeting-limiter-api")
	if err := run(logger); err != nil {
		logger.ErrorContext(context.Background(), "server stopped", "err", err)
		os.Exit(1)
	}
}

func run(logger *slog.Logger) error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	dir, err := people.NewDirectory(people.Seed())
	if err != nil {
		return fmt.Errorf("load people: %w", err)
	}
	store, err := calendar.NewFileStore(cfg.DataFile)
	if err != nil {
		return fmt.Errorf("open store: %w", err)
	}
	svc := calendar.NewService(store, dir, time.Local, rand.Text)
	srv := &http.Server{
		Addr:              fmt.Sprintf("127.0.0.1:%d", cfg.Port),
		Handler:           httpapi.New(svc, dir, logger),
		ReadHeaderTimeout: 5 * time.Second,
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	errs := make(chan error, 1)
	go func() {
		logger.InfoContext(ctx, "listening", "addr", srv.Addr)
		errs <- srv.ListenAndServe()
	}()
	select {
	case err := <-errs:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return srv.Shutdown(shutdown)
	}
}
