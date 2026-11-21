package calendar

import (
	"cmp"
	"context"
	"slices"
	"time"

	"meetinglimiter/internal/money"
	"meetinglimiter/internal/people"
)

type Spender struct {
	User     people.User `json:"user"`
	Spent    money.Cents `json:"spent"`
	Minutes  int         `json:"minutes"`
	Meetings int         `json:"meetings"`
}

type Rankings struct {
	Meetings  []Meeting `json:"meetings"`
	Spenders  []Spender `json:"spenders"`
	TimeSinks []Spender `json:"timeSinks"`
}

func (s *Service) Rankings(ctx context.Context, from, to time.Time, limit int) (Rankings, error) {
	meetings, err := s.Meetings(ctx, from, to)
	if err != nil {
		return Rankings{}, err
	}
	byCost := slices.Clone(meetings)
	slices.SortStableFunc(byCost, func(a, b Meeting) int { return cmp.Compare(b.TotalCost, a.TotalCost) })
	var spenders []Spender
	for _, u := range s.people.All() {
		sp := Spender{User: u}
		for _, m := range meetings {
			if c := m.ChargeFor(u.ID); c > 0 {
				sp.Spent += c
				sp.Minutes += m.Minutes()
				sp.Meetings++
			}
		}
		if sp.Meetings > 0 {
			spenders = append(spenders, sp)
		}
	}
	bySpent := slices.Clone(spenders)
	slices.SortStableFunc(bySpent, func(a, b Spender) int { return cmp.Compare(b.Spent, a.Spent) })
	byTime := slices.Clone(spenders)
	slices.SortStableFunc(byTime, func(a, b Spender) int { return cmp.Compare(b.Minutes, a.Minutes) })
	return Rankings{
		Meetings:  top(byCost, limit),
		Spenders:  top(bySpent, limit),
		TimeSinks: top(byTime, limit),
	}, nil
}

func top[T any](s []T, n int) []T {
	if len(s) > n {
		s = s[:n]
	}
	if s == nil {
		return []T{}
	}
	return s
}
