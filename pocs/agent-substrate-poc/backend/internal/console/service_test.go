package console

import (
	"context"
	"errors"
	"testing"

	"agentsubstratepoc/internal/agent"
	"agentsubstratepoc/internal/ate"
)

type fakeStore struct {
	ops []string
}

func (f *fakeStore) ListActors(context.Context) ([]ate.Actor, error) { return nil, nil }
func (f *fakeStore) GetActor(_ context.Context, n string) (ate.Actor, error) {
	f.ops = append(f.ops, "get "+n)
	return ate.Actor{Name: n, State: ate.StateRunning}, nil
}
func (f *fakeStore) CreateActor(_ context.Context, n string) error {
	f.ops = append(f.ops, "create "+n)
	return nil
}
func (f *fakeStore) SuspendActor(_ context.Context, n string) error {
	f.ops = append(f.ops, "suspend "+n)
	return nil
}
func (f *fakeStore) ResumeActor(_ context.Context, n string) error {
	f.ops = append(f.ops, "resume "+n)
	return nil
}
func (f *fakeStore) DeleteActor(_ context.Context, n string) error {
	f.ops = append(f.ops, "delete "+n)
	return nil
}
func (f *fakeStore) ListWorkers(context.Context) ([]ate.Worker, error) { return nil, nil }

type fakeProxies struct {
	closed []string
}

func (p *fakeProxies) Port(context.Context, string) (int, error) { return 18100, nil }
func (p *fakeProxies) Close(a string)                            { p.closed = append(p.closed, a) }

type fakeAgent struct{ port int }

func (a *fakeAgent) State(_ context.Context, port int) (agent.State, error) {
	a.port = port
	return agent.State{BootID: "b"}, nil
}
func (a *fakeAgent) AddNote(_ context.Context, port int, text string) (agent.Note, error) {
	a.port = port
	return agent.Note{ID: 1, Text: text}, nil
}

func TestInvalidNamesNeverReachTheCluster(t *testing.T) {
	store := &fakeStore{}
	s := NewService(store, &fakeProxies{}, &fakeAgent{})
	for _, name := range []string{"", "Alice", "a_b", "-a", "a-", "x;rm -rf"} {
		if _, err := s.Create(context.Background(), name); !errors.Is(err, ErrInvalid) {
			t.Errorf("%q: err = %v, want ErrInvalid", name, err)
		}
	}
	if len(store.ops) != 0 {
		t.Errorf("store was called: %v", store.ops)
	}
}

func TestCreateReturnsTheFreshActor(t *testing.T) {
	store := &fakeStore{}
	a, err := NewService(store, &fakeProxies{}, &fakeAgent{}).Create(context.Background(), "alice")
	if err != nil {
		t.Fatal(err)
	}
	if a.Name != "alice" || len(store.ops) != 2 || store.ops[0] != "create alice" {
		t.Errorf("ops = %v", store.ops)
	}
}

func TestDeleteClosesTheProxyBeforeDeleting(t *testing.T) {
	store, p := &fakeStore{}, &fakeProxies{}
	if err := NewService(store, p, &fakeAgent{}).Delete(context.Background(), "alice"); err != nil {
		t.Fatal(err)
	}
	if len(p.closed) != 1 || store.ops[0] != "delete alice" {
		t.Errorf("closed=%v ops=%v", p.closed, store.ops)
	}
}

func TestNotesGoThroughTheActorProxy(t *testing.T) {
	a := &fakeAgent{}
	note, err := NewService(&fakeStore{}, &fakeProxies{}, a).Remember(context.Background(), "alice", "hello")
	if err != nil {
		t.Fatal(err)
	}
	if a.port != 18100 || note.Text != "hello" {
		t.Errorf("port=%d note=%+v", a.port, note)
	}
}

func TestEmptyNoteIsInvalid(t *testing.T) {
	_, err := NewService(&fakeStore{}, &fakeProxies{}, &fakeAgent{}).Remember(context.Background(), "alice", "")
	if !errors.Is(err, ErrInvalid) {
		t.Errorf("err = %v", err)
	}
}
