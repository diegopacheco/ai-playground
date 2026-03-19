package agents

import "os/exec"

type CodexBuilder struct {
	Model string
}

func (b *CodexBuilder) BuildCommand(prompt string) *exec.Cmd {
	return exec.Command("codex", "exec", "--full-auto", "-m", b.Model, prompt)
}
