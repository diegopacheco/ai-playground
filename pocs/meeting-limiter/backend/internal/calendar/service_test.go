package calendar

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"meetinglimiter/internal/money"
	"meetinglimiter/internal/people"
)

var monday = time.Date(2026, 11, 16, 9, 0, 0, 0, time.UTC)

func newService(t *testing.T) *Service {
	t.Helper()
	dir, err := people.NewDirectory(people.Seed())
	if err != nil {
		t.Fatal(err)
	}
	store, err := NewFileStore("")
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	return NewService(store, dir, time.UTC, func() string { n++; return fmt.Sprintf("m%d", n) })
}

func req(organizer string, start time.Time, hours float64, attendees ...string) Request {
	return Request{
		Title:       "sync",
		OrganizerID: organizer,
		AttendeeIDs: attendees,
		Start:       start,
		End:         start.Add(time.Duration(hours * float64(time.Hour))),
	}
}

func book(t *testing.T, s *Service, r Request) Meeting {
	t.Helper()
	m, err := s.Book(context.Background(), r)
	if err != nil {
		t.Fatalf("book %+v: %v", r, err)
	}
	return m
}

func balanceOf(t *testing.T, s *Service, at time.Time, id string) money.Cents {
	t.Helper()
	ws, err := s.Wallets(context.Background(), at)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range ws {
		if w.User.ID == id {
			return w.Balance
		}
	}
	t.Fatalf("no wallet for %s", id)
	return 0
}

func TestWeeklyHourCapPerRole(t *testing.T) {
	tests := []struct {
		name    string
		user    string
		capHour int
	}{
		{"engineers can spend 5 hours of meetings a week", "ana", 5},
		{"managers can spend 10 hours of meetings a week", "daniel", 10},
		{"directors can spend 15 hours of meetings a week", "fabio", 15},
		{"vps can spend 15 hours of meetings a week", "gabriela", 15},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := newService(t)
			for h := 0; h < tt.capHour; h++ {
				day := h / 8
				book(t, s, req(tt.user, monday.AddDate(0, 0, day).Add(time.Duration(h%8)*time.Hour), 1))
			}
			if got := balanceOf(t, s, monday, tt.user); got != 0 {
				t.Fatalf("balance after %dh = %v, want $0.00", tt.capHour, got)
			}
			_, err := s.Book(context.Background(), req(tt.user, monday.AddDate(0, 0, 3), 0.25))
			var short *InsufficientFundsError
			if !errors.As(err, &short) {
				t.Fatalf("one more slot over the cap must be refused, got %v", err)
			}
		})
	}
}

func TestSixHoursRuinsAnEngineerWeek(t *testing.T) {
	s := newService(t)
	book(t, s, req("ana", monday, 3))
	book(t, s, req("ana", monday.Add(4*time.Hour), 2))
	_, err := s.Book(context.Background(), req("ana", monday.AddDate(0, 0, 1), 1))
	var short *InsufficientFundsError
	if !errors.As(err, &short) {
		t.Fatalf("6th hour must be refused, got %v", err)
	}
	if len(short.Quote.Short()) != 1 || short.Quote.Short()[0].BalanceAfter != -10000 {
		t.Fatalf("quote must show ana short by $100, got %+v", short.Quote.Short())
	}
}

func TestWalletRefillsNextWeek(t *testing.T) {
	s := newService(t)
	book(t, s, req("ana", monday, 5))
	nextMonday := monday.AddDate(0, 0, 7)
	if got := balanceOf(t, s, nextMonday, "ana"); got != 50000 {
		t.Fatalf("next week balance = %v, want full $500.00", got)
	}
	book(t, s, req("ana", nextMonday, 1))
	if got := balanceOf(t, s, monday, "ana"); got != 0 {
		t.Fatalf("booking next week must not touch this week, got %v", got)
	}
}

func TestEveryParticipantPaysAndAnyShortBlocksBooking(t *testing.T) {
	s := newService(t)
	book(t, s, req("bruno", monday, 5))
	_, err := s.Book(context.Background(), req("gabriela", monday.AddDate(0, 0, 1), 1, "bruno", "daniel"))
	var short *InsufficientFundsError
	if !errors.As(err, &short) {
		t.Fatalf("a broke attendee must block the meeting, got %v", err)
	}
	if got := balanceOf(t, s, monday, "gabriela"); got != 375000 {
		t.Fatalf("refused booking must not charge the organizer, got %v", got)
	}
	m := book(t, s, req("gabriela", monday.AddDate(0, 0, 1), 1, "daniel", "ana"))
	want := money.Cents(25000 + 15000 + 10000)
	if m.TotalCost != want {
		t.Fatalf("meeting cost = %v, want %v (vp + manager + engineer for 1h)", m.TotalCost, want)
	}
	if got := balanceOf(t, s, monday, "daniel"); got != 150000-15000 {
		t.Fatalf("attendee must be charged, balance %v", got)
	}
}

func TestDuplicateAttendeesPayOnce(t *testing.T) {
	s := newService(t)
	m := book(t, s, req("ana", monday, 1, "ana", "bruno", "bruno"))
	if len(m.Charges) != 2 || m.TotalCost != 20000 {
		t.Fatalf("charges = %+v, want ana and bruno once each", m.Charges)
	}
}

func TestCancelRefundsEveryone(t *testing.T) {
	s := newService(t)
	m := book(t, s, req("ana", monday, 5, "daniel"))
	if err := s.Cancel(context.Background(), m.ID); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"ana", "daniel"} {
		ws, _ := s.Wallets(context.Background(), monday)
		for _, w := range ws {
			if w.User.ID == id && w.Spent != 0 {
				t.Fatalf("%s still charged %v after cancel", id, w.Spent)
			}
		}
	}
	if err := s.Cancel(context.Background(), m.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("second cancel = %v, want ErrNotFound", err)
	}
}

func TestQuoteValidation(t *testing.T) {
	sunday := time.Date(2026, 11, 22, 23, 0, 0, 0, time.UTC)
	tests := []struct {
		name string
		r    Request
	}{
		{"title is required", Request{OrganizerID: "ana", Start: monday, End: monday.Add(time.Hour)}},
		{"end before start", req("ana", monday, -1)},
		{"duration in 15 minute steps", req("ana", monday, 0.1)},
		{"longer than 8 hours", req("ana", monday, 8.25)},
		{"crossing into next week", req("ana", sunday, 2)},
		{"unknown organizer", req("nobody", monday, 1)},
		{"unknown attendee", req("ana", monday, 1, "ghost")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newService(t).Quote(context.Background(), tt.r)
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("got %v, want ErrInvalid", err)
			}
		})
	}
}

func TestWeekStartsMonday(t *testing.T) {
	tests := []struct {
		name string
		in   time.Time
		want time.Time
	}{
		{"monday morning is its own week", monday, time.Date(2026, 11, 16, 0, 0, 0, 0, time.UTC)},
		{"sunday night belongs to the week that began monday", time.Date(2026, 11, 22, 23, 59, 0, 0, time.UTC), time.Date(2026, 11, 16, 0, 0, 0, 0, time.UTC)},
		{"next monday midnight starts a new week", time.Date(2026, 11, 23, 0, 0, 0, 0, time.UTC), time.Date(2026, 11, 23, 0, 0, 0, 0, time.UTC)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := WeekStart(tt.in, time.UTC); !got.Equal(tt.want) {
				t.Fatalf("WeekStart(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestRankings(t *testing.T) {
	s := newService(t)
	book(t, s, req("ana", monday, 1))
	big := book(t, s, req("gabriela", monday.Add(2*time.Hour), 2, "fabio", "daniel"))
	book(t, s, req("bruno", monday.AddDate(0, 0, 1), 4))
	r, err := s.Rankings(context.Background(), monday.AddDate(0, 0, -1), monday.AddDate(0, 0, 7), 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Meetings) != 2 || r.Meetings[0].ID != big.ID {
		t.Fatalf("most expensive meeting first, got %+v", r.Meetings)
	}
	if r.Spenders[0].User.ID == "bruno" {
		t.Fatalf("money rank must not follow time: bruno spent $400 while leaders spent $500, got %+v", r.Spenders)
	}
	if r.TimeSinks[0].User.ID != "bruno" || r.TimeSinks[0].Minutes != 240 {
		t.Fatalf("bruno spent the most time (4h), got %+v", r.TimeSinks)
	}
	if len(r.Spenders) != 2 {
		t.Fatalf("limit must cap the list, got %d", len(r.Spenders))
	}
}

func TestFileStorePersistsAcrossRestarts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "meetings.json")
	first, err := NewFileStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Add(context.Background(), Meeting{ID: "a", Title: "kept", Start: monday, End: monday.Add(time.Hour)}); err != nil {
		t.Fatal(err)
	}
	second, err := NewFileStore(path)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := second.List(context.Background())
	if len(got) != 1 || got[0].Title != "kept" {
		t.Fatalf("restart lost meetings: %+v", got)
	}
}
