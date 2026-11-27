package proxy

import (
	"context"
	"errors"
	"testing"
)

type fakeProc struct {
	alive   bool
	stopped bool
}

func (p *fakeProc) Stop()       { p.stopped = true; p.alive = false }
func (p *fakeProc) Alive() bool { return p.alive }

type fakeStarter struct {
	procs []*fakeProc
	args  [][]string
}

func (s *fakeStarter) Start(args []string) (Process, error) {
	p := &fakeProc{alive: true}
	s.procs = append(s.procs, p)
	s.args = append(s.args, args)
	return p, nil
}

func ok(context.Context, int) error { return nil }

func TestPortIsReusedWhileTheProxyIsAlive(t *testing.T) {
	s := &fakeStarter{}
	m := NewManager(s, ok, "poc", 18100, 5)
	a, _ := m.Port(context.Background(), "alice")
	b, _ := m.Port(context.Background(), "alice")
	if a != b || len(s.procs) != 1 {
		t.Errorf("one proxy per actor: ports %d %d, starts %d", a, b, len(s.procs))
	}
	if got := s.args[0]; got[2] != "alice" || got[3] != "18100:80" || got[5] != "poc" {
		t.Errorf("proxy args = %v", got)
	}
}

func TestEachActorGetsItsOwnPort(t *testing.T) {
	m := NewManager(&fakeStarter{}, ok, "poc", 18100, 5)
	a, _ := m.Port(context.Background(), "alice")
	b, _ := m.Port(context.Background(), "bob")
	if a == b {
		t.Errorf("two actors share port %d", a)
	}
}

func TestDeadProxyIsRestarted(t *testing.T) {
	s := &fakeStarter{}
	m := NewManager(s, ok, "poc", 18100, 5)
	if _, err := m.Port(context.Background(), "alice"); err != nil {
		t.Fatal(err)
	}
	s.procs[0].alive = false
	if _, err := m.Port(context.Background(), "alice"); err != nil {
		t.Fatal(err)
	}
	if len(s.procs) != 2 {
		t.Errorf("a crashed proxy must be replaced, starts = %d", len(s.procs))
	}
}

func TestPortRangeExhaustion(t *testing.T) {
	m := NewManager(&fakeStarter{}, ok, "poc", 18100, 1)
	if _, err := m.Port(context.Background(), "alice"); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Port(context.Background(), "bob"); !errors.Is(err, ErrNoFreePort) {
		t.Errorf("err = %v, want ErrNoFreePort", err)
	}
}

func TestProxyThatNeverListensIsStopped(t *testing.T) {
	s := &fakeStarter{}
	fail := func(context.Context, int) error { return errors.New("timeout") }
	m := NewManager(s, fail, "poc", 18100, 2)
	if _, err := m.Port(context.Background(), "alice"); err == nil {
		t.Fatal("want error")
	}
	if !s.procs[0].stopped {
		t.Error("a proxy that never opened its port must be killed")
	}
}

func TestCloseFreesThePort(t *testing.T) {
	s := &fakeStarter{}
	m := NewManager(s, ok, "poc", 18100, 1)
	if _, err := m.Port(context.Background(), "alice"); err != nil {
		t.Fatal(err)
	}
	m.Close("alice")
	if !s.procs[0].stopped {
		t.Error("close must stop the process")
	}
	if _, err := m.Port(context.Background(), "bob"); err != nil {
		t.Errorf("port should be free after close: %v", err)
	}
}
