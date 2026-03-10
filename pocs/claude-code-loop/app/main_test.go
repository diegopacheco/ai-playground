package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"
)

func newTestServer() *Server {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	return NewServer(logger)
}

func TestNewServer(t *testing.T) {
	srv := newTestServer()
	if srv == nil {
		t.Fatal("expected server to be non-nil")
	}
	if len(srv.stocks) != 8 {
		t.Fatalf("expected 8 stocks, got %d", len(srv.stocks))
	}
	if srv.mux == nil {
		t.Fatal("expected mux to be non-nil")
	}
	if srv.logger == nil {
		t.Fatal("expected logger to be non-nil")
	}
}

func TestGetStocks(t *testing.T) {
	srv := newTestServer()
	stocks := srv.GetStocks()
	if len(stocks) != 8 {
		t.Fatalf("expected 8 stocks, got %d", len(stocks))
	}
	stocks[0].Price = 999999
	original := srv.GetStocks()
	if original[0].Price == 999999 {
		t.Fatal("GetStocks should return a copy, not a reference")
	}
}

func TestAPIHandler(t *testing.T) {
	srv := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/api/stocks", nil)
	w := httptest.NewRecorder()
	srv.APIHandler(w, req)
	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if ct != "application/json" {
		t.Fatalf("expected application/json, got %s", ct)
	}
	cors := resp.Header.Get("Access-Control-Allow-Origin")
	if cors != "*" {
		t.Fatalf("expected CORS header *, got %s", cors)
	}
	var stocks []Stock
	if err := json.NewDecoder(resp.Body).Decode(&stocks); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if len(stocks) != 8 {
		t.Fatalf("expected 8 stocks, got %d", len(stocks))
	}
}

func TestAPIHandlerCORSPreflight(t *testing.T) {
	srv := newTestServer()
	req := httptest.NewRequest(http.MethodOptions, "/api/stocks", nil)
	w := httptest.NewRecorder()
	srv.APIHandler(w, req)
	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 on OPTIONS, got %d", resp.StatusCode)
	}
	if resp.Header.Get("Access-Control-Allow-Methods") != "GET, OPTIONS" {
		t.Fatal("missing Access-Control-Allow-Methods header")
	}
}

func TestDashboardHandler(t *testing.T) {
	srv := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	srv.DashboardHandler(w, req)
	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if ct != "text/html" {
		t.Fatalf("expected text/html, got %s", ct)
	}
	body := w.Body.String()
	if len(body) == 0 {
		t.Fatal("expected non-empty HTML body")
	}
}

func TestSimulatePricesChangesValues(t *testing.T) {
	srv := newTestServer()
	original := srv.GetStocks()
	ctx, cancel := context.WithCancel(context.Background())
	go srv.SimulatePrices(ctx)
	time.Sleep(3 * time.Second)
	cancel()
	updated := srv.GetStocks()
	changed := false
	for i := range original {
		if original[i].Price != updated[i].Price {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("expected at least one stock price to change after simulation")
	}
}

func TestSimulatePricesCancellation(t *testing.T) {
	srv := newTestServer()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		srv.SimulatePrices(ctx)
		close(done)
	}()
	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("SimulatePrices did not stop after context cancellation")
	}
}

func TestStockPriceFloor(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	srv := &Server{
		stocks: []Stock{{Symbol: "TEST", Price: 0.5, Change: 0.0}},
		mux:    http.NewServeMux(),
		logger: logger,
	}
	ctx, cancel := context.WithCancel(context.Background())
	go srv.SimulatePrices(ctx)
	time.Sleep(3 * time.Second)
	cancel()
	stocks := srv.GetStocks()
	if stocks[0].Price < 1 {
		t.Fatalf("expected price >= 1, got %f", stocks[0].Price)
	}
}

func TestStockSymbols(t *testing.T) {
	srv := newTestServer()
	expected := []string{"AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"}
	stocks := srv.GetStocks()
	for i, s := range stocks {
		if s.Symbol != expected[i] {
			t.Fatalf("expected symbol %s at index %d, got %s", expected[i], i, s.Symbol)
		}
	}
}
