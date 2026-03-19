package agents

import "os/exec"

type ClaudeBuilder struct {
	Model string
}

func (b *ClaudeBuilder) BuildCommand(prompt string) *exec.Cmd {
	return exec.Command("claude", "-p", prompt, "--model", b.Model, "--dangerously-skip-permissions")
}
