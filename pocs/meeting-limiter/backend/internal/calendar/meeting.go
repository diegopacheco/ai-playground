package calendar

import (
	"errors"
	"fmt"
	"time"

	"meetinglimiter/internal/money"
	"meetinglimiter/internal/people"
)

var (
	ErrInvalid  = errors.New("invalid meeting")
	ErrNotFound = errors.New("meeting not found")
)

type InsufficientFundsError struct {
	Quote Quote
}

func (e *InsufficientFundsError) Error() string {
	return fmt.Sprintf("insufficient funds for %d participant(s)", len(e.Quote.Short()))
}

const (
	SlotMinutes    = 15
	MaxMinutes     = 8 * 60
	MaxTitleLength = 120
)

type Charge struct {
	UserID string      `json:"userId"`
	Amount money.Cents `json:"amount"`
}

type Meeting struct {
	ID          string      `json:"id"`
	Title       string      `json:"title"`
	OrganizerID string      `json:"organizerId"`
	AttendeeIDs []string    `json:"attendeeIds"`
	Start       time.Time   `json:"start"`
	End         time.Time   `json:"end"`
	Charges     []Charge    `json:"charges"`
	TotalCost   money.Cents `json:"totalCost"`
}

func (m Meeting) Minutes() int {
	return int(m.End.Sub(m.Start) / time.Minute)
}

func (m Meeting) ChargeFor(userID string) money.Cents {
	for _, c := range m.Charges {
		if c.UserID == userID {
			return c.Amount
		}
	}
	return 0
}

type Request struct {
	Title       string    `json:"title"`
	OrganizerID string    `json:"organizerId"`
	AttendeeIDs []string  `json:"attendeeIds"`
	Start       time.Time `json:"start"`
	End         time.Time `json:"end"`
}

func (r Request) Participants() []string {
	ids := []string{r.OrganizerID}
	seen := map[string]bool{r.OrganizerID: true}
	for _, id := range r.AttendeeIDs {
		if !seen[id] {
			seen[id] = true
			ids = append(ids, id)
		}
	}
	return ids
}

type Line struct {
	User          people.User `json:"user"`
	Cost          money.Cents `json:"cost"`
	BalanceBefore money.Cents `json:"balanceBefore"`
	BalanceAfter  money.Cents `json:"balanceAfter"`
	Allowed       bool        `json:"allowed"`
}

type Quote struct {
	WeekStart time.Time   `json:"weekStart"`
	Minutes   int         `json:"minutes"`
	Lines     []Line      `json:"lines"`
	TotalCost money.Cents `json:"totalCost"`
	Allowed   bool        `json:"allowed"`
}

func (q Quote) Short() []Line {
	var short []Line
	for _, l := range q.Lines {
		if !l.Allowed {
			short = append(short, l)
		}
	}
	return short
}

type Wallet struct {
	User         people.User `json:"user"`
	WeekStart    time.Time   `json:"weekStart"`
	Budget       money.Cents `json:"budget"`
	Spent        money.Cents `json:"spent"`
	Balance      money.Cents `json:"balance"`
	MinutesUsed  int         `json:"minutesUsed"`
	MinutesLimit int         `json:"minutesLimit"`
	Meetings     int         `json:"meetings"`
}

func WeekStart(t time.Time, loc *time.Location) time.Time {
	local := t.In(loc)
	offset := (int(local.Weekday()) + 6) % 7
	day := time.Date(local.Year(), local.Month(), local.Day(), 0, 0, 0, 0, loc)
	return day.AddDate(0, 0, -offset)
}

func (r Request) validate(loc *time.Location) error {
	switch {
	case r.Title == "":
		return fmt.Errorf("%w: title is required", ErrInvalid)
	case len([]rune(r.Title)) > MaxTitleLength:
		return fmt.Errorf("%w: title longer than %d characters", ErrInvalid, MaxTitleLength)
	case r.OrganizerID == "":
		return fmt.Errorf("%w: organizer is required", ErrInvalid)
	case r.Start.IsZero() || r.End.IsZero():
		return fmt.Errorf("%w: start and end are required", ErrInvalid)
	case !r.End.After(r.Start):
		return fmt.Errorf("%w: end must be after start", ErrInvalid)
	}
	d := r.End.Sub(r.Start)
	if d%(SlotMinutes*time.Minute) != 0 {
		return fmt.Errorf("%w: duration must be in %d minute steps", ErrInvalid, SlotMinutes)
	}
	if d > MaxMinutes*time.Minute {
		return fmt.Errorf("%w: meetings last at most %d hours", ErrInvalid, MaxMinutes/60)
	}
	if !WeekStart(r.Start, loc).Equal(WeekStart(r.End.Add(-time.Nanosecond), loc)) {
		return fmt.Errorf("%w: a meeting cannot cross into the next week", ErrInvalid)
	}
	return nil
}
