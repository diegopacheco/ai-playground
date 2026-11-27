package ate

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"
)

var (
	ErrNotFound      = errors.New("not found")
	ErrAlreadyExists = errors.New("already exists")
	ErrConflict      = errors.New("conflict with current actor state")
)

type CommandError struct {
	Args   []string
	Output string
	kind   error
}

func (e *CommandError) Error() string {
	return fmt.Sprintf("kubectl-ate %s: %s", strings.Join(e.Args, " "), strings.TrimSpace(e.Output))
}

func (e *CommandError) Unwrap() error {
	return e.kind
}

func classify(output string) error {
	switch {
	case strings.Contains(output, "NotFound") || strings.Contains(output, "not found"):
		return ErrNotFound
	case strings.Contains(output, "AlreadyExists") || strings.Contains(output, "already exists"):
		return ErrAlreadyExists
	case strings.Contains(output, "FailedPrecondition"):
		return ErrConflict
	default:
		return nil
	}
}

type Runner interface {
	Run(ctx context.Context, args ...string) ([]byte, error)
}

type CLI struct {
	Bin     string
	Context string
}

func (c CLI) Run(ctx context.Context, args ...string) ([]byte, error) {
	full := append([]string{"--context", c.Context}, args...)
	cmd := exec.CommandContext(ctx, c.Bin, full...)
	var out, errOut bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errOut
	if err := cmd.Run(); err != nil {
		msg := errOut.String() + out.String()
		return nil, &CommandError{Args: args, Output: msg, kind: classify(msg)}
	}
	return out.Bytes(), nil
}

type Client struct {
	run      Runner
	atespace string
	template string
}

func NewClient(run Runner, atespace, template string) *Client {
	return &Client{run: run, atespace: atespace, template: template}
}

func (c *Client) Atespace() string {
	return c.atespace
}

func (c *Client) ListActors(ctx context.Context) ([]Actor, error) {
	out, err := c.run.Run(ctx, "get", "actors", "-a", c.atespace, "-o", "json")
	if err != nil {
		return nil, fmt.Errorf("list actors: %w", err)
	}
	var wire wireActors
	if err := json.Unmarshal(out, &wire); err != nil {
		return nil, fmt.Errorf("decode actors: %w", err)
	}
	actors := make([]Actor, 0, len(wire.Actors))
	for _, w := range wire.Actors {
		if w.ActorTemplate.Name == c.template && w.Metadata.Name != "" && !isGolden(w.Metadata.Name) {
			actors = append(actors, w.toActor())
		}
	}
	return actors, nil
}

func isGolden(name string) bool {
	return len(name) == 36 && strings.Count(name, "-") == 4
}

func (c *Client) GetActor(ctx context.Context, name string) (Actor, error) {
	out, err := c.run.Run(ctx, "get", "actor", name, "-a", c.atespace, "-o", "json")
	if err != nil {
		return Actor{}, fmt.Errorf("get actor %s: %w", name, err)
	}
	var w wireActor
	if err := json.Unmarshal(out, &w); err != nil {
		return Actor{}, fmt.Errorf("decode actor %s: %w", name, err)
	}
	return w.toActor(), nil
}

func (c *Client) CreateActor(ctx context.Context, name string) error {
	_, err := c.run.Run(ctx, "create", "actor", name, "--template", c.template, "-a", c.atespace)
	if err != nil {
		return fmt.Errorf("create actor %s: %w", name, err)
	}
	return nil
}

func (c *Client) SuspendActor(ctx context.Context, name string) error {
	return c.verb(ctx, "suspend", name)
}

func (c *Client) ResumeActor(ctx context.Context, name string) error {
	return c.verb(ctx, "resume", name)
}

func (c *Client) DeleteActor(ctx context.Context, name string) error {
	return c.verb(ctx, "delete", name)
}

func (c *Client) verb(ctx context.Context, verb, name string) error {
	if _, err := c.run.Run(ctx, verb, "actor", name, "-a", c.atespace); err != nil {
		return fmt.Errorf("%s actor %s: %w", verb, name, err)
	}
	return nil
}

func (c *Client) ListWorkers(ctx context.Context) ([]Worker, error) {
	out, err := c.run.Run(ctx, "get", "workers", "-o", "json")
	if err != nil {
		return nil, fmt.Errorf("list workers: %w", err)
	}
	var wire wireWorkers
	if err := json.Unmarshal(out, &wire); err != nil {
		return nil, fmt.Errorf("decode workers: %w", err)
	}
	workers := make([]Worker, 0, len(wire.Workers))
	for _, w := range wire.Workers {
		workers = append(workers, w.toWorker())
	}
	return workers, nil
}
