package agents

type AgentInfo struct {
	Name   string   `json:"name"`
	Models []string `json:"models"`
}

var AvailableAgents = []AgentInfo{
	{Name: "claude", Models: []string{"opus", "sonnet", "haiku"}},
	{Name: "gemini", Models: []string{"gemini-3.1-pro", "gemini-3-flash", "gemini-2.5-pro"}},
	{Name: "copilot", Models: []string{"claude-sonnet-4.6", "claude-sonnet-4.5", "gemini-3-pro"}},
	{Name: "codex", Models: []string{"gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"}},
}

func GetRunner(name string, model string) *AgentRunner {
	var builder CLIBuilder
	switch name {
	case "claude":
		builder = &ClaudeBuilder{Model: model}
	case "gemini":
		builder = &GeminiBuilder{}
	case "copilot":
		builder = &CopilotBuilder{Model: model}
	case "codex":
		builder = &CodexBuilder{Model: model}
	default:
		return nil
	}
	return &AgentRunner{Name: name, Model: model, Builder: builder}
}
