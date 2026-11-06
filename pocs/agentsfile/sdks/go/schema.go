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
	attrs:   map[string]valueKind{"description": valueString, "source": valueString, "content": valueString},
}

var skillFileSchema = schema{
	context: "file",
	attrs:   map[string]valueKind{"source": valueString, "content": valueString},
}

var agentSchema = schema{
	context: "agent",
	attrs: map[string]valueKind{
		"description": valueString,
		"model":       valueString,
		"tools":       valueList,
		"source":      valueString,
		"content":     valueString,
	},
}

var commandSchema = schema{
	context: "command",
	attrs: map[string]valueKind{
		"description":   valueString,
		"argument-hint": valueString,
		"model":         valueString,
		"tools":         valueList,
		"source":        valueString,
		"content":       valueString,
	},
}

var mcpSchema = schema{
	context: "mcp",
	attrs: map[string]valueKind{
		"description": valueString,
		"source":      valueString,
		"transport":   valueString,
		"command":     valueString,
		"args":        valueList,
		"env":         valueMap,
		"url":         valueString,
		"headers":     valueMap,
	},
	check: checkTransport,
}

var pluginSchema = schema{
	context: "plugin",
	attrs: map[string]valueKind{
		"description": valueString,
		"version":     valueString,
		"author":      valueString,
		"source":      valueString,
	},
}

var profileKinds = []string{"skill", "agent", "command", "mcp", "plugin"}

var pluginKinds = []string{"skill", "agent", "command", "mcp"}

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

func duplicate(s statement, owner statement) *Error {
	return newError(s.line, s.col, "duplicate %s \"%s\" in %s \"%s\"", s.name, s.label, owner.name, owner.label)
}

func exactlyOne(block statement, hasSource, hasInline bool, inline string) error {
	if !hasSource && !hasInline {
		return newError(block.line, block.col, "%s \"%s\" requires \"source\" or %s", block.name, block.label, inline)
	}
	if hasSource && hasInline {
		return newError(block.line, block.col, "%s \"%s\" cannot set both \"source\" and %s", block.name, block.label, inline)
	}
	return nil
}

func sourceOrContent(block statement, attrs attributes) error {
	return exactlyOne(block, attrs.has("source"), attrs.has("content"), "\"content\"")
}

func (a attributes) has(name string) bool {
	_, ok := a[name]
	return ok
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

type items struct {
	skills   []Skill
	agents   []Agent
	commands []Command
	mcps     []Mcp
	plugins  []Plugin
}

func (it items) empty() bool {
	return len(it.skills)+len(it.agents)+len(it.commands)+len(it.mcps)+len(it.plugins) == 0
}

func (it *items) add(s statement) error {
	switch s.name {
	case "skill":
		item, err := buildSkill(s)
		it.skills = append(it.skills, item)
		return err
	case "agent":
		item, err := buildAgent(s)
		it.agents = append(it.agents, item)
		return err
	case "command":
		item, err := buildCommand(s)
		it.commands = append(it.commands, item)
		return err
	case "mcp":
		item, err := buildMcp(s)
		it.mcps = append(it.mcps, item)
		return err
	default:
		item, err := buildPlugin(s)
		it.plugins = append(it.plugins, item)
		return err
	}
}

func buildContainer(owner statement, sc schema, kinds []string) (attributes, items, error) {
	it := items{skills: []Skill{}, agents: []Agent{}, commands: []Command{}, mcps: []Mcp{}, plugins: []Plugin{}}
	seen := map[string]map[string]bool{}
	for _, kind := range kinds {
		seen[kind] = map[string]bool{}
	}
	onBlock := func(s statement) error {
		names, known := seen[s.name]
		if !known {
			return unknownBlock(s, sc.context)
		}
		if names[s.label] {
			return duplicate(s, owner)
		}
		names[s.label] = true
		return it.add(s)
	}
	attrs, err := walk(sc, owner.body, onBlock)
	return attrs, it, err
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
	attrs, it, err := buildContainer(block, profileSchema, profileKinds)
	if err != nil {
		return Profile{}, err
	}
	return Profile{
		Name:        block.label,
		Description: attrs.optional("description"),
		Skills:      it.skills,
		Agents:      it.agents,
		Commands:    it.commands,
		Mcps:        it.mcps,
		Plugins:     it.plugins,
	}, nil
}

func buildPlugin(block statement) (Plugin, error) {
	attrs, it, err := buildContainer(block, pluginSchema, pluginKinds)
	if err != nil {
		return Plugin{}, err
	}
	if err := exactlyOne(block, attrs.has("source"), !it.empty(), "inline items"); err != nil {
		return Plugin{}, err
	}
	return Plugin{
		Name:        block.label,
		Description: attrs.optional("description"),
		Version:     attrs.optional("version"),
		Author:      attrs.optional("author"),
		Source:      attrs.optional("source"),
		Skills:      it.skills,
		Agents:      it.agents,
		Commands:    it.commands,
		Mcps:        it.mcps,
	}, nil
}

func buildSkill(block statement) (Skill, error) {
	files := []SkillFile{}
	paths := map[string]bool{}
	onBlock := func(s statement) error {
		if s.name != "file" {
			return unknownBlock(s, skillSchema.context)
		}
		if paths[s.label] {
			return duplicate(s, block)
		}
		paths[s.label] = true
		file, err := buildSkillFile(s)
		files = append(files, file)
		return err
	}
	attrs, err := walk(skillSchema, block.body, onBlock)
	if err != nil {
		return Skill{}, err
	}
	if err := sourceOrContent(block, attrs); err != nil {
		return Skill{}, err
	}
	return Skill{
		Name:        block.label,
		Description: attrs.optional("description"),
		Source:      attrs.optional("source"),
		Content:     attrs.optional("content"),
		Files:       files,
	}, nil
}

func buildSkillFile(block statement) (SkillFile, error) {
	attrs, err := walk(skillFileSchema, block.body, nil)
	if err != nil {
		return SkillFile{}, err
	}
	if err := sourceOrContent(block, attrs); err != nil {
		return SkillFile{}, err
	}
	return SkillFile{Path: block.label, Source: attrs.optional("source"), Content: attrs.optional("content")}, nil
}

func buildAgent(block statement) (Agent, error) {
	attrs, err := walk(agentSchema, block.body, nil)
	if err != nil {
		return Agent{}, err
	}
	if err := sourceOrContent(block, attrs); err != nil {
		return Agent{}, err
	}
	return Agent{
		Name:        block.label,
		Description: attrs.optional("description"),
		Model:       attrs.optional("model"),
		Tools:       attrs.list("tools"),
		Source:      attrs.optional("source"),
		Content:     attrs.optional("content"),
	}, nil
}

func buildCommand(block statement) (Command, error) {
	attrs, err := walk(commandSchema, block.body, nil)
	if err != nil {
		return Command{}, err
	}
	if err := sourceOrContent(block, attrs); err != nil {
		return Command{}, err
	}
	return Command{
		Name:         block.label,
		Description:  attrs.optional("description"),
		ArgumentHint: attrs.optional("argument-hint"),
		Model:        attrs.optional("model"),
		Tools:        attrs.list("tools"),
		Source:       attrs.optional("source"),
		Content:      attrs.optional("content"),
	}, nil
}

func buildMcp(block statement) (Mcp, error) {
	attrs, err := walk(mcpSchema, block.body, nil)
	if err != nil {
		return Mcp{}, err
	}
	if err := exactlyOne(block, attrs.has("source"), attrs.has("transport"), "\"transport\""); err != nil {
		return Mcp{}, err
	}
	var transport *Transport
	if s, ok := attrs["transport"]; ok {
		t := Transport(s.value.str)
		transport = &t
		required := transportRequirement[t]
		if !attrs.has(required) {
			return Mcp{}, newError(block.line, block.col, "mcp \"%s\" with %s transport requires \"%s\"", block.label, t, required)
		}
	}
	return Mcp{
		Name:        block.label,
		Description: attrs.optional("description"),
		Source:      attrs.optional("source"),
		Transport:   transport,
		Command:     attrs.optional("command"),
		Args:        attrs.list("args"),
		Env:         attrs.mapping("env"),
		URL:         attrs.optional("url"),
		Headers:     attrs.mapping("headers"),
	}, nil
}
