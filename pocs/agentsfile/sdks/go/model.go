package agentsfile

type Transport string

const (
	Stdio Transport = "stdio"
	HTTP  Transport = "http"
	SSE   Transport = "sse"
)

type Agentsfile struct {
	Version  string
	Profiles []Profile
}

type Profile struct {
	Name        string
	Description *string
	Skills      []Skill
	Agents      []Agent
	Commands    []Command
	Mcps        []Mcp
	Plugins     []Plugin
}

type Skill struct {
	Name        string
	Description *string
	Source      *string
	Content     *string
	Files       []SkillFile
}

type SkillFile struct {
	Path    string
	Source  *string
	Content *string
}

type Agent struct {
	Name        string
	Description *string
	Model       *string
	Tools       []string
	Source      *string
	Content     *string
}

type Command struct {
	Name         string
	Description  *string
	ArgumentHint *string
	Model        *string
	Tools        []string
	Source       *string
	Content      *string
}

type Mcp struct {
	Name        string
	Description *string
	Source      *string
	Transport   *Transport
	Command     *string
	Args        []string
	Env         map[string]string
	URL         *string
	Headers     map[string]string
}

type Plugin struct {
	Name        string
	Description *string
	Version     *string
	Author      *string
	Source      *string
	Skills      []Skill
	Agents      []Agent
	Commands    []Command
	Mcps        []Mcp
}
