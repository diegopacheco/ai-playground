package calendar

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"time"

	"meetinglimiter/internal/money"
	"meetinglimiter/internal/people"
)

type store interface {
	List(ctx context.Context) ([]Meeting, error)
	Add(ctx context.Context, m Meeting) error
	Delete(ctx context.Context, id string) error
}

type directory interface {
	Get(id string) (people.User, error)
	All() []people.User
}

type Service struct {
	mu     sync.Mutex
	store  store
	people directory
	loc    *time.Location
	newID  func() string
}

func NewService(s store, d directory, loc *time.Location, newID func() string) *Service {
	return &Service{store: s, people: d, loc: loc, newID: newID}
}

func (s *Service) Meetings(ctx context.Context, from, to time.Time) ([]Meeting, error) {
	all, err := s.store.List(ctx)
	if err != nil {
		return nil, fmt.Errorf("list meetings: %w", err)
	}
	var out []Meeting
	for _, m := range all {
		if !m.Start.Before(from) && m.Start.Before(to) {
			out = append(out, m)
		}
	}
	slices.SortFunc(out, func(a, b Meeting) int { return a.Start.Compare(b.Start) })
	return out, nil
}

func (s *Service) Wallets(ctx context.Context, at time.Time) ([]Wallet, error) {
	week := WeekStart(at, s.loc)
	meetings, err := s.weekMeetings(ctx, week)
	if err != nil {
		return nil, err
	}
	var out []Wallet
	for _, u := range s.people.All() {
		w, err := s.wallet(u, week, meetings)
		if err != nil {
			return nil, err
		}
		out = append(out, w)
	}
	return out, nil
}

func (s *Service) Quote(ctx context.Context, r Request) (Quote, error) {
	if err := r.validate(s.loc); err != nil {
		return Quote{}, err
	}
	week := WeekStart(r.Start, s.loc)
	meetings, err := s.weekMeetings(ctx, week)
	if err != nil {
		return Quote{}, err
	}
	minutes := int(r.End.Sub(r.Start) / time.Minute)
	q := Quote{WeekStart: week, Minutes: minutes, Allowed: true}
	for _, id := range r.Participants() {
		u, err := s.people.Get(id)
		if err != nil {
			return Quote{}, fmt.Errorf("%w: %w", ErrInvalid, err)
		}
		w, err := s.wallet(u, week, meetings)
		if err != nil {
			return Quote{}, err
		}
		p, err := people.PolicyFor(u.Role)
		if err != nil {
			return Quote{}, err
		}
		cost := money.PerMinutes(p.HourlyRate, minutes)
		after := w.Balance - cost
		line := Line{User: u, Cost: cost, BalanceBefore: w.Balance, BalanceAfter: after, Allowed: after >= 0}
		q.Lines = append(q.Lines, line)
		q.TotalCost += cost
		q.Allowed = q.Allowed && line.Allowed
	}
	return q, nil
}

func (s *Service) Book(ctx context.Context, r Request) (Meeting, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	q, err := s.Quote(ctx, r)
	if err != nil {
		return Meeting{}, err
	}
	if !q.Allowed {
		return Meeting{}, &InsufficientFundsError{Quote: q}
	}
	m := Meeting{
		ID:          s.newID(),
		Title:       r.Title,
		OrganizerID: r.OrganizerID,
		AttendeeIDs: r.Participants()[1:],
		Start:       r.Start,
		End:         r.End,
		TotalCost:   q.TotalCost,
	}
	for _, l := range q.Lines {
		m.Charges = append(m.Charges, Charge{UserID: l.User.ID, Amount: l.Cost})
	}
	if err := s.store.Add(ctx, m); err != nil {
		return Meeting{}, fmt.Errorf("save meeting: %w", err)
	}
	return m, nil
}

func (s *Service) Cancel(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.store.Delete(ctx, id)
}

func (s *Service) weekMeetings(ctx context.Context, week time.Time) ([]Meeting, error) {
	return s.Meetings(ctx, week, week.AddDate(0, 0, 7))
}

func (s *Service) wallet(u people.User, week time.Time, meetings []Meeting) (Wallet, error) {
	p, err := people.PolicyFor(u.Role)
	if err != nil {
		return Wallet{}, err
	}
	w := Wallet{User: u, WeekStart: week, Budget: p.WeeklyBudget(), MinutesLimit: p.WeeklyHours * 60}
	for _, m := range meetings {
		if c := m.ChargeFor(u.ID); c > 0 {
			w.Spent += c
			w.MinutesUsed += m.Minutes()
			w.Meetings++
		}
	}
	w.Balance = w.Budget - w.Spent
	return w, nil
}
