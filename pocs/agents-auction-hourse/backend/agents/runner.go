package agents

import (
	"context"
	"os/exec"
	"time"
)

type CLIBuilder interface {
	BuildCommand(prompt string) *exec.Cmd
}

type AgentRunner struct {
	Name    string
	Model   string
	Builder CLIBuilder
}

func (r *AgentRunner) Run(prompt string) (string, int64, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := r.Builder.BuildCommand(prompt)
	cmd = exec.CommandContext(ctx, cmd.Path, cmd.Args[1:]...)

	start := time.Now()
	output, err := cmd.CombinedOutput()
	elapsed := time.Since(start).Milliseconds()

	return string(output), elapsed, err
}
