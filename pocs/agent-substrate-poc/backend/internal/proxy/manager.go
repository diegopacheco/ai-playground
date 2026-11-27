package proxy

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os/exec"
	"strconv"
	"sync"
	"time"
)

var ErrNoFreePort = errors.New("no free proxy port")

type Process interface {
	Stop()
	Alive() bool
}

type Starter interface {
	Start(args []string) (Process, error)
}

type PortWaiter func(ctx context.Context, port int) error

type entry struct {
	port int
	proc Process
}

type Manager struct {
	mu       sync.Mutex
	atespace string
	base     int
	count    int
	start    Starter
	wait     PortWaiter
	entries  map[string]entry
}

func NewManager(start Starter, wait PortWaiter, atespace string, base, count int) *Manager {
	return &Manager{atespace: atespace, base: base, count: count, start: start, wait: wait, entries: map[string]entry{}}
}

func (m *Manager) Port(ctx context.Context, actor string) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e, ok := m.entries[actor]; ok {
		if e.proc.Alive() {
			return e.port, nil
		}
		delete(m.entries, actor)
	}
	port, err := m.freePort()
	if err != nil {
		return 0, err
	}
	proc, err := m.start.Start([]string{"proxy", "actor", actor, strconv.Itoa(port) + ":80", "-a", m.atespace})
	if err != nil {
		return 0, fmt.Errorf("start proxy for %s: %w", actor, err)
	}
	if err := m.wait(ctx, port); err != nil {
		proc.Stop()
		return 0, fmt.Errorf("proxy for %s on port %d: %w", actor, port, err)
	}
	m.entries[actor] = entry{port: port, proc: proc}
	return port, nil
}

func (m *Manager) freePort() (int, error) {
	used := map[int]bool{}
	for _, e := range m.entries {
		used[e.port] = true
	}
	for p := m.base; p < m.base+m.count; p++ {
		if !used[p] {
			return p, nil
		}
	}
	return 0, ErrNoFreePort
}

func (m *Manager) Close(actor string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e, ok := m.entries[actor]; ok {
		e.proc.Stop()
		delete(m.entries, actor)
	}
}

func (m *Manager) CloseAll() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for name, e := range m.entries {
		e.proc.Stop()
		delete(m.entries, name)
	}
}

type ExecStarter struct {
	Bin     string
	Context string
}

type execProcess struct {
	cmd  *exec.Cmd
	done chan struct{}
}

func (p *execProcess) Stop() {
	if p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
	}
	<-p.done
}

func (p *execProcess) Alive() bool {
	select {
	case <-p.done:
		return false
	default:
		return true
	}
}

func (s ExecStarter) Start(args []string) (Process, error) {
	cmd := exec.Command(s.Bin, append([]string{"--context", s.Context}, args...)...)
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	p := &execProcess{cmd: cmd, done: make(chan struct{})}
	go func() {
		_ = cmd.Wait()
		close(p.done)
	}()
	return p, nil
}

func WaitTCP(ctx context.Context, port int) error {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	addr := net.JoinHostPort("127.0.0.1", strconv.Itoa(port))
	for {
		conn, err := (&net.Dialer{Timeout: time.Second}).DialContext(ctx, "tcp", addr)
		if err == nil {
			return conn.Close()
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("port %d never opened: %w", port, ctx.Err())
		case <-time.After(200 * time.Millisecond):
		}
	}
}
