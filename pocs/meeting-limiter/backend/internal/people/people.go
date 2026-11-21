package people

import (
	"errors"
	"fmt"

	"meetinglimiter/internal/money"
)

type Role string

const (
	Engineer Role = "engineer"
	Manager  Role = "manager"
	Director Role = "director"
	VP       Role = "vp"
)

var ErrUnknownUser = errors.New("unknown user")

type Policy struct {
	Role        Role        `json:"role"`
	WeeklyHours int         `json:"weeklyHours"`
	HourlyRate  money.Cents `json:"hourlyRate"`
}

func (p Policy) WeeklyBudget() money.Cents {
	return p.HourlyRate * money.Cents(p.WeeklyHours)
}

var policies = []Policy{
	{Role: Engineer, WeeklyHours: 5, HourlyRate: 10000},
	{Role: Manager, WeeklyHours: 10, HourlyRate: 15000},
	{Role: Director, WeeklyHours: 15, HourlyRate: 25000},
	{Role: VP, WeeklyHours: 15, HourlyRate: 25000},
}

func Policies() []Policy {
	return append([]Policy(nil), policies...)
}

func PolicyFor(r Role) (Policy, error) {
	for _, p := range policies {
		if p.Role == r {
			return p, nil
		}
	}
	return Policy{}, fmt.Errorf("no policy for role %q", r)
}

type User struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Title string `json:"title"`
	Role  Role   `json:"role"`
}

type Directory struct {
	users []User
}

func NewDirectory(users []User) (*Directory, error) {
	seen := map[string]bool{}
	for _, u := range users {
		if seen[u.ID] {
			return nil, fmt.Errorf("duplicate user id %s", u.ID)
		}
		if _, err := PolicyFor(u.Role); err != nil {
			return nil, fmt.Errorf("user %s: %w", u.ID, err)
		}
		seen[u.ID] = true
	}
	return &Directory{users: append([]User(nil), users...)}, nil
}

func (d *Directory) All() []User {
	return append([]User(nil), d.users...)
}

func (d *Directory) Get(id string) (User, error) {
	for _, u := range d.users {
		if u.ID == id {
			return u, nil
		}
	}
	return User{}, fmt.Errorf("%w: %s", ErrUnknownUser, id)
}

func Seed() []User {
	return []User{
		{ID: "ana", Name: "Ana Souza", Title: "Software Engineer", Role: Engineer},
		{ID: "bruno", Name: "Bruno Lima", Title: "Software Engineer", Role: Engineer},
		{ID: "carla", Name: "Carla Mendes", Title: "Senior Engineer", Role: Engineer},
		{ID: "hugo", Name: "Hugo Alves", Title: "Staff Engineer", Role: Engineer},
		{ID: "daniel", Name: "Daniel Rocha", Title: "Engineering Manager", Role: Manager},
		{ID: "elisa", Name: "Elisa Prado", Title: "Product Manager", Role: Manager},
		{ID: "fabio", Name: "Fabio Nunes", Title: "Director of Engineering", Role: Director},
		{ID: "gabriela", Name: "Gabriela Torres", Title: "VP of Engineering", Role: VP},
	}
}
