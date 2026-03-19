package agents

import "os/exec"

type CopilotBuilder struct {
	Model string
}

func (b *CopilotBuilder) BuildCommand(prompt string) *exec.Cmd {
	return exec.Command("copilot", "--allow-all", "--model", b.Model, "-p", prompt)
}
