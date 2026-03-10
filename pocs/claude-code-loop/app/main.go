package main

import (
	"context"
	"embed"
	"encoding/json"
	"log/slog"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

//go:embed static/index.html
var staticFiles embed.FS

type Stock struct {
	Symbol string  `json:"symbol"`
	Price  float64 `json:"price"`
	Change float64 `json:"change"`
}

type Server struct {
	stocks []Stock
	mu     sync.RWMutex
	mux    *http.ServeMux
	logger *slog.Logger
}

func NewServer(logger *slog.Logger) *Server {
	s := &Server{
		stocks: []Stock{
			{Symbol: "AAPL", Price: 178.50, Change: 0.0},
			{Symbol: "GOOGL", Price: 141.20, Change: 0.0},
			{Symbol: "MSFT", Price: 415.30, Change: 0.0},
			{Symbol: "AMZN", Price: 185.60, Change: 0.0},
			{Symbol: "TSLA", Price: 248.90, Change: 0.0},
			{Symbol: "NVDA", Price: 875.40, Change: 0.0},
			{Symbol: "META", Price: 505.75, Change: 0.0},
			{Symbol: "NFLX", Price: 612.30, Change: 0.0},
		},
		mux:    http.NewServeMux(),
		logger: logger,
	}
	s.mux.HandleFunc("/", s.DashboardHandler)
	s.mux.HandleFunc("/api/stocks", s.APIHandler)
	return s
}

func (s *Server) SimulatePrices(ctx context.Context) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			s.logger.Info("price simulator stopped")
			return
		case <-ticker.C:
			s.mu.Lock()
			for i := range s.stocks {
				delta := (rand.Float64() - 0.50) * 5.0
				s.stocks[i].Change = delta
				s.stocks[i].Price += delta
				if s.stocks[i].Price < 1 {
					s.stocks[i].Price = 1
				}
			}
			s.mu.Unlock()
		}
	}
}

func (s *Server) GetStocks() []Stock {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]Stock, len(s.stocks))
	copy(result, s.stocks)
	return result
}

func (s *Server) APIHandler(w http.ResponseWriter, r *http.Request) {
	s.logger.Info("api request", "method", r.Method, "path", r.URL.Path)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	stocks := s.GetStocks()
	data, err := json.Marshal(stocks)
	if err != nil {
		s.logger.Error("failed to marshal stocks", "error", err)
		http.Error(w, `{"error":"internal server error"}`, http.StatusInternalServerError)
		return
	}
	w.Write(data)
}

func (s *Server) DashboardHandler(w http.ResponseWriter, r *http.Request) {
	s.logger.Info("dashboard request", "method", r.Method, "path", r.URL.Path)
	data, err := staticFiles.ReadFile("static/index.html")
	if err != nil {
		s.logger.Error("failed to read index.html", "error", err)
		http.Error(w, "internal server error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html")
	w.Write(data)
}

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	srv := NewServer(logger)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go srv.SimulatePrices(ctx)

	httpServer := &http.Server{
		Addr:    ":" + port,
		Handler: srv.mux,
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		logger.Info("shutting down")
		cancel()
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		httpServer.Shutdown(shutdownCtx)
	}()

	logger.Info("server starting", "port", port)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		logger.Error("server failed", "error", err)
		os.Exit(1)
	}
	logger.Info("server stopped")
}
