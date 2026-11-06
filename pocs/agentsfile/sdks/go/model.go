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
}

type Skill struct {
	Name        string
	Source      string
	Description *string
}

type Agent struct {
	Name        string
	Source      string
	Description *string
	Model       *string
	Tools       []string
}

type Command struct {
	Name        string
	Source      string
	Description *string
}

type Mcp struct {
	Name        string
	Transport   Transport
	Description *string
	Command     *string
	Args        []string
	Env         map[string]string
	URL         *string
	Headers     map[string]string
}
