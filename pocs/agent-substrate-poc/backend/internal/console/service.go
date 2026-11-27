package console

import (
	"context"
	"errors"
	"fmt"
	"regexp"

	"agentsubstratepoc/internal/agent"
	"agentsubstratepoc/internal/ate"
)

var ErrInvalid = errors.New("invalid input")

var dnsLabel = regexp.MustCompile(`^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$`)

type actorStore interface {
	ListActors(ctx context.Context) ([]ate.Actor, error)
	GetActor(ctx context.Context, name string) (ate.Actor, error)
	CreateActor(ctx context.Context, name string) error
	SuspendActor(ctx context.Context, name string) error
	ResumeActor(ctx context.Context, name string) error
	DeleteActor(ctx context.Context, name string) error
	ListWorkers(ctx context.Context) ([]ate.Worker, error)
}

type proxies interface {
	Port(ctx context.Context, actor string) (int, error)
	Close(actor string)
}

type agentClient interface {
	State(ctx context.Context, port int) (agent.State, error)
	AddNote(ctx context.Context, port int, text string) (agent.Note, error)
}

type Service struct {
	actors  actorStore
	proxies proxies
	agent   agentClient
}

func NewService(actors actorStore, p proxies, a agentClient) *Service {
	return &Service{actors: actors, proxies: p, agent: a}
}

func validName(name string) error {
	if !dnsLabel.MatchString(name) {
		return fmt.Errorf("%w: actor name %q must be a DNS-1123 label", ErrInvalid, name)
	}
	return nil
}

func (s *Service) Actors(ctx context.Context) ([]ate.Actor, error) {
	return s.actors.ListActors(ctx)
}

func (s *Service) Workers(ctx context.Context) ([]ate.Worker, error) {
	return s.actors.ListWorkers(ctx)
}

func (s *Service) Actor(ctx context.Context, name string) (ate.Actor, error) {
	if err := validName(name); err != nil {
		return ate.Actor{}, err
	}
	return s.actors.GetActor(ctx, name)
}

func (s *Service) Create(ctx context.Context, name string) (ate.Actor, error) {
	if err := validName(name); err != nil {
		return ate.Actor{}, err
	}
	if err := s.actors.CreateActor(ctx, name); err != nil {
		return ate.Actor{}, err
	}
	return s.actors.GetActor(ctx, name)
}

func (s *Service) Suspend(ctx context.Context, name string) (ate.Actor, error) {
	return s.transition(ctx, name, s.actors.SuspendActor)
}

func (s *Service) Resume(ctx context.Context, name string) (ate.Actor, error) {
	return s.transition(ctx, name, s.actors.ResumeActor)
}

func (s *Service) transition(ctx context.Context, name string, op func(context.Context, string) error) (ate.Actor, error) {
	if err := validName(name); err != nil {
		return ate.Actor{}, err
	}
	if err := op(ctx, name); err != nil {
		return ate.Actor{}, err
	}
	return s.actors.GetActor(ctx, name)
}

func (s *Service) Delete(ctx context.Context, name string) error {
	if err := validName(name); err != nil {
		return err
	}
	s.proxies.Close(name)
	return s.actors.DeleteActor(ctx, name)
}

func (s *Service) State(ctx context.Context, name string) (agent.State, error) {
	if err := validName(name); err != nil {
		return agent.State{}, err
	}
	port, err := s.proxies.Port(ctx, name)
	if err != nil {
		return agent.State{}, err
	}
	return s.agent.State(ctx, port)
}

func (s *Service) Remember(ctx context.Context, name, text string) (agent.Note, error) {
	if err := validName(name); err != nil {
		return agent.Note{}, err
	}
	if text == "" {
		return agent.Note{}, fmt.Errorf("%w: note text is empty", ErrInvalid)
	}
	port, err := s.proxies.Port(ctx, name)
	if err != nil {
		return agent.Note{}, err
	}
	return s.agent.AddNote(ctx, port, text)
}
