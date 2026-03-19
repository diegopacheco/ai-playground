package agents

import "os/exec"

type GeminiBuilder struct{}

func (b *GeminiBuilder) BuildCommand(prompt string) *exec.Cmd {
	return exec.Command("gemini", "-y", "-p", prompt)
}
