package main

import (
	"context"
	"crypto/rand"
	"errors"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"agentsubstratepoc/internal/agent"
	"agentsubstratepoc/internal/logx"
)

func main() {
	logger := logx.New("agent")
	journal := os.Getenv("JOURNAL_PATH")
	if journal == "" {
		journal = "/data/journal.log"
	}
	addr := os.Getenv("AGENT_ADDR")
	if addr == "" {
		addr = ":80"
	}
	hostname, err := os.Hostname()
	if err != nil {
		hostname = "unknown"
	}
	memory := agent.NewMemory(rand.Text()[:10], hostname, journal, time.Now)
	server := &http.Server{Addr: addr, Handler: logx.Middleware(agent.Routes(memory, logger)), ReadHeaderTimeout: 5 * time.Second}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, os.Interrupt)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdown); err != nil {
			logger.Error("shutdown", "err", err)
		}
	}()
	logger.Info("agent listening", "addr", addr, "journal", journal)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logger.Error("serve", "err", err)
		os.Exit(1)
	}
}
