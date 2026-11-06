package agentsfile

type schema struct {
	context string
	attrs   map[string]valueKind
	check   func(statement) error
}

var fileSchema = schema{
	context: "file",
	attrs:   map[string]valueKind{"version": valueString},
	check:   checkVersion,
}

var profileSchema = schema{
	context: "profile",
	attrs:   map[string]valueKind{"description": valueString},
}

var skillSchema = schema{
	context: "skill",
	attrs:   map[string]valueKind{"source": valueString, "description": valueString},
}

var agentSchema = schema{
	context: "agent",
	attrs: map[string]valueKind{
		"source":      valueString,
		"description": valueString,
		"model":       valueString,
		"tools":       valueList,
	},
}

var commandSchema = schema{
	context: "command",
	attrs:   map[string]valueKind{"source": valueString, "description": valueString},
}

var mcpSchema = schema{
	context: "mcp",
	attrs: map[string]valueKind{
		"transport":   valueString,
		"description": valueString,
		"command":     valueString,
		"args":        valueList,
		"env":         valueMap,
		"url":         valueString,
		"headers":     valueMap,
	},
	check: checkTransport,
}

var transportRequirement = map[Transport]string{
	Stdio: "command",
	HTTP:  "url",
	SSE:   "url",
}

type attributes map[string]statement

func checkVersion(s statement) error {
	if s.name == "version" && s.value.str != "1" {
		return newError(s.value.line, s.value.col, "unsupported version \"%s\"", s.value.str)
	}
	return nil
}

func checkTransport(s statement) error {
	if s.name != "transport" {
		return nil
	}
	if _, ok := transportRequirement[Transport(s.value.str)]; !ok {
		return newError(s.value.line, s.value.col, "invalid transport \"%s\", expected stdio, http or sse", s.value.str)
	}
	return nil
}

func walk(sc schema, body []statement, onBlock func(statement) error) (attributes, error) {
	attrs := attributes{}
	for _, s := range body {
		if s.isBlock {
			if onBlock == nil {
				return nil, unknownBlock(s, sc.context)
			}
			if err := onBlock(s); err != nil {
				return nil, err
			}
			continue
		}
		if err := checkAttribute(sc, attrs, s); err != nil {
			return nil, err
		}
	}
	return attrs, nil
}

func checkAttribute(sc schema, attrs attributes, s statement) error {
	kind, known := sc.attrs[s.name]
	if !known {
		return newError(s.line, s.col, "unknown attribute \"%s\" in %s", s.name, sc.context)
	}
	if _, duplicated := attrs[s.name]; duplicated {
		return newError(s.line, s.col, "duplicate attribute \"%s\" in %s", s.name, sc.context)
	}
	if s.value.kind != kind {
		return newError(s.line, s.col, "attribute \"%s\" must be %s", s.name, valueKindNames[kind])
	}
	attrs[s.name] = s
	if sc.check != nil {
		return sc.check(s)
	}
	return nil
}

func unknownBlock(s statement, context string) *Error {
	return newError(s.line, s.col, "unknown block \"%s\" in %s", s.name, context)
}

func (a attributes) require(block statement, name string) (string, error) {
	s, ok := a[name]
	if !ok {
		return "", newError(block.line, block.col, "missing required attribute \"%s\" in %s \"%s\"", name, block.name, block.label)
	}
	return s.value.str, nil
}

func (a attributes) optional(name string) *string {
	s, ok := a[name]
	if !ok {
		return nil
	}
	text := s.value.str
	return &text
}

func (a attributes) list(name string) []string {
	if s, ok := a[name]; ok {
		return s.value.list
	}
	return []string{}
}

func (a attributes) mapping(name string) map[string]string {
	if s, ok := a[name]; ok {
		return s.value.entries
	}
	return map[string]string{}
}

func buildFile(stmts []statement) (*Agentsfile, error) {
	file := &Agentsfile{Profiles: []Profile{}}
	names := map[string]bool{}
	onBlock := func(s statement) error {
		if s.name != "profile" {
			return unknownBlock(s, fileSchema.context)
		}
		if names[s.label] {
			return newError(s.line, s.col, "duplicate profile \"%s\"", s.label)
		}
		names[s.label] = true
		profile, err := buildProfile(s)
		if err != nil {
			return err
		}
		file.Profiles = append(file.Profiles, profile)
		return nil
	}
	attrs, err := walk(fileSchema, stmts, onBlock)
	if err != nil {
		return nil, err
	}
	version, ok := attrs["version"]
	if !ok {
		return nil, newError(1, 1, "missing required attribute \"version\"")
	}
	file.Version = version.value.str
	return file, nil
}

func buildProfile(block statement) (Profile, error) {
	p := Profile{
		Name:     block.label,
		Skills:   []Skill{},
		Agents:   []Agent{},
		Commands: []Command{},
		Mcps:     []Mcp{},
	}
	seen := map[string]map[string]bool{"skill": {}, "agent": {}, "command": {}, "mcp": {}}
	onBlock := func(s statement) error {
		names, known := seen[s.name]
		if !known {
			return unknownBlock(s, profileSchema.context)
		}
		if names[s.label] {
			return newError(s.line, s.col, "duplicate %s \"%s\" in profile \"%s\"", s.name, s.label, block.label)
		}
		names[s.label] = true
		return p.add(s)
	}
	attrs, err := walk(profileSchema, block.body, onBlock)
	if err != nil {
		return p, err
	}
	p.Description = attrs.optional("description")
	return p, nil
}

func (p *Profile) add(s statement) error {
	switch s.name {
	case "skill":
		item, err := buildSkill(s)
		p.Skills = append(p.Skills, item)
		return err
	case "agent":
		item, err := buildAgent(s)
		p.Agents = append(p.Agents, item)
		return err
	case "command":
		item, err := buildCommand(s)
		p.Commands = append(p.Commands, item)
		return err
	default:
		item, err := buildMcp(s)
		p.Mcps = append(p.Mcps, item)
		return err
	}
}

func buildSkill(block statement) (Skill, error) {
	attrs, err := walk(skillSchema, block.body, nil)
	if err != nil {
		return Skill{}, err
	}
	source, err := attrs.require(block, "source")
	return Skill{Name: block.label, Source: source, Description: attrs.optional("description")}, err
}

func buildAgent(block statement) (Agent, error) {
	attrs, err := walk(agentSchema, block.body, nil)
	if err != nil {
		return Agent{}, err
	}
	source, err := attrs.require(block, "source")
	return Agent{
		Name:        block.label,
		Source:      source,
		Description: attrs.optional("description"),
		Model:       attrs.optional("model"),
		Tools:       attrs.list("tools"),
	}, err
}

func buildCommand(block statement) (Command, error) {
	attrs, err := walk(commandSchema, block.body, nil)
	if err != nil {
		return Command{}, err
	}
	source, err := attrs.require(block, "source")
	return Command{Name: block.label, Source: source, Description: attrs.optional("description")}, err
}

func buildMcp(block statement) (Mcp, error) {
	attrs, err := walk(mcpSchema, block.body, nil)
	if err != nil {
		return Mcp{}, err
	}
	transport, err := attrs.require(block, "transport")
	if err != nil {
		return Mcp{}, err
	}
	required := transportRequirement[Transport(transport)]
	if _, ok := attrs[required]; !ok {
		return Mcp{}, newError(block.line, block.col, "mcp \"%s\" with %s transport requires \"%s\"", block.label, transport, required)
	}
	return Mcp{
		Name:        block.label,
		Transport:   Transport(transport),
		Description: attrs.optional("description"),
		Command:     attrs.optional("command"),
		Args:        attrs.list("args"),
		Env:         attrs.mapping("env"),
		URL:         attrs.optional("url"),
		Headers:     attrs.mapping("headers"),
	}, nil
}
