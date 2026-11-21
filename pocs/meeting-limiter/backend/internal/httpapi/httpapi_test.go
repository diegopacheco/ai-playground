package httpapi

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"meetinglimiter/internal/calendar"
	"meetinglimiter/internal/people"
)

func server(t *testing.T) *httptest.Server {
	t.Helper()
	dir, err := people.NewDirectory(people.Seed())
	if err != nil {
		t.Fatal(err)
	}
	store, err := calendar.NewFileStore("")
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	svc := calendar.NewService(store, dir, time.UTC, func() string { n++; return "m" + strings.Repeat("x", n) })
	srv := httptest.NewServer(New(svc, dir, slog.New(slog.NewTextHandler(io.Discard, nil))))
	t.Cleanup(srv.Close)
	return srv
}

func call(t *testing.T, method, url, body string) *http.Response {
	t.Helper()
	r, err := http.NewRequest(method, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	res, err := http.DefaultClient.Do(r)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { res.Body.Close() })
	return res
}

const fiveHours = `{"title":"planning","organizerId":"ana","attendeeIds":[],"start":"2026-11-16T09:00:00Z","end":"2026-11-16T14:00:00Z"}`
const oneMore = `{"title":"one more","organizerId":"daniel","attendeeIds":["ana"],"start":"2026-11-17T09:00:00Z","end":"2026-11-17T10:00:00Z"}`

func TestBookingStatusCodes(t *testing.T) {
	srv := server(t)
	if res := call(t, "POST", srv.URL+"/api/meetings", fiveHours); res.StatusCode != http.StatusCreated {
		t.Fatalf("first booking status %d, want 201", res.StatusCode)
	}
	res := call(t, "POST", srv.URL+"/api/meetings", oneMore)
	if res.StatusCode != http.StatusPaymentRequired {
		t.Fatalf("over budget status %d, want 402 so the UI can explain who is short", res.StatusCode)
	}
	var body shortBody
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	short := body.Quote.Short()
	if len(short) != 1 || short[0].User.ID != "ana" {
		t.Fatalf("402 must name ana as short, got %+v", short)
	}
	if res := call(t, "POST", srv.URL+"/api/meetings", `{"title":""}`); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("invalid status %d, want 400", res.StatusCode)
	}
	if res := call(t, "DELETE", srv.URL+"/api/meetings/missing", ""); res.StatusCode != http.StatusNotFound {
		t.Fatalf("missing delete status %d, want 404", res.StatusCode)
	}
}

func TestQuoteDoesNotCharge(t *testing.T) {
	srv := server(t)
	for range 3 {
		if res := call(t, "POST", srv.URL+"/api/meetings/quote", fiveHours); res.StatusCode != http.StatusOK {
			t.Fatalf("quote status %d", res.StatusCode)
		}
	}
	res := call(t, "GET", srv.URL+"/api/wallets?week=2026-11-18T00:00:00Z", "")
	var wallets []calendar.Wallet
	if err := json.NewDecoder(res.Body).Decode(&wallets); err != nil {
		t.Fatal(err)
	}
	for _, w := range wallets {
		if w.Spent != 0 {
			t.Fatalf("quotes must never spend money, %s spent %v", w.User.ID, w.Spent)
		}
	}
}

func TestRankingsRejectsBadLimit(t *testing.T) {
	srv := server(t)
	base := srv.URL + "/api/rankings?from=2026-11-16T00:00:00Z&to=2026-11-23T00:00:00Z"
	if res := call(t, "GET", base+"&limit=0", ""); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("limit 0 status %d, want 400", res.StatusCode)
	}
	if res := call(t, "GET", base+"&limit=5", ""); res.StatusCode != http.StatusOK {
		t.Fatalf("limit 5 status %d, want 200", res.StatusCode)
	}
}

func TestCorrelationIdEchoed(t *testing.T) {
	srv := server(t)
	r, _ := http.NewRequest("GET", srv.URL+"/api/health", nil)
	r.Header.Set("X-Correlation-Id", "abc")
	res, err := http.DefaultClient.Do(r)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.Header.Get("X-Correlation-Id") != "abc" {
		t.Fatal("correlation id must be echoed so UI and API logs can be joined")
	}
}
