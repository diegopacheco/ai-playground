package main

import (
	"context"
	"errors"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"agentsubstratepoc/internal/actorclient"
	"agentsubstratepoc/internal/ate"
	"agentsubstratepoc/internal/config"
	"agentsubstratepoc/internal/console"
	"agentsubstratepoc/internal/httpapi"
	"agentsubstratepoc/internal/logx"
	"agentsubstratepoc/internal/proxy"
)

func main() {
	logger := logx.New("console")
	cfg, err := config.FromEnv()
	if err != nil {
		logger.Error("config", "err", err)
		os.Exit(1)
	}
	cli := ate.CLI{Bin: cfg.AteBin, Context: cfg.KubeContext}
	proxies := proxy.NewManager(proxy.ExecStarter{Bin: cfg.AteBin, Context: cfg.KubeContext}, proxy.WaitTCP, cfg.Atespace, cfg.ProxyPortBase, cfg.ProxyPortCount)
	defer proxies.CloseAll()
	svc := console.NewService(ate.NewClient(cli, cfg.Atespace, cfg.Template), proxies, actorclient.New("127.0.0.1", 90*time.Second))
	server := &http.Server{
		Addr:              "127.0.0.1:" + strconv.Itoa(cfg.Port),
		Handler:           httpapi.New(svc, logger),
		ReadHeaderTimeout: 5 * time.Second,
	}
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
	logger.Info("console listening", "addr", server.Addr, "atespace", cfg.Atespace, "template", cfg.Template)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logger.Error("serve", "err", err)
		os.Exit(1)
	}
}
