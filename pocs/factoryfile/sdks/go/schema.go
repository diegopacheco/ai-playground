package factoryfile

import (
	"regexp"
	"strconv"
)

type schema struct {
	context string
	attrs   map[string]valueKind
	refs    map[string]string
	check   func(statement) error
}

var fileSchema = schema{
	context: "file",
	attrs:   map[string]valueKind{"version": valueString},
	check:   checkVersion,
}

var factorySchema = schema{
	context: "factory",
	attrs:   map[string]valueKind{"description": valueString, "entry": valueString},
	refs:    map[string]string{"entry": "stage"},
}

var agentSchema = schema{
	context: "agent",
	attrs: map[string]valueKind{
		"description": valueString,
		"role":        valueString,
		"model":       valueString,
		"tools":       valueList,
		"source":      valueString,
		"content":     valueString,
	},
}

var stageSchema = schema{
	context: "stage",
	attrs: map[string]valueKind{
		"description": valueString,
		"agents":      valueList,
		"next":        valueList,
		"artifacts":   valueList,
		"content":     valueString,
	},
	refs: map[string]string{"agents": "agent", "next": "stage"},
}

var loopSchema = schema{
	context: "loop",
	attrs: map[string]valueKind{
		"description": valueString,
		"from":        valueString,
		"to":          valueString,
		"until":       valueString,
		"max":         valueString,
	},
	refs:  map[string]string{"from": "stage", "to": "stage"},
	check: checkMax,
}

var maxPattern = regexp.MustCompile(`^[1-9][0-9]{0,5}$`)

type attributes map[string]statement

type reference struct {
	target string
	names  []string
	owner  string
	line   int
	col    int
}

func checkVersion(s statement) error {
	if s.name == "version" && s.value.str != "1" {
		return newError(s.value.line, s.value.col, "unsupported version \"%s\"", s.value.str)
	}
	return nil
}

func checkMax(s statement) error {
	if s.name == "max" && !maxPattern.MatchString(s.value.str) {
		return newError(s.value.line, s.value.col, "invalid max \"%s\", expected an integer from 1 to 999999", s.value.str)
	}
	return nil
}

func walk(sc schema, block statement, refs *[]reference, onBlock func(statement) error) (attributes, error) {
	attrs := attributes{}
	for _, s := range block.body {
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
		collectReference(sc, block, refs, s)
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

func collectReference(sc schema, block statement, refs *[]reference, s statement) {
	target, ok := sc.refs[s.name]
	if !ok {
		return
	}
	names := s.value.list
	if s.value.kind == valueString {
		names = []string{s.value.str}
	}
	*refs = append(*refs, reference{
		target: target,
		names:  names,
		owner:  sc.context + " \"" + block.label + "\"",
		line:   s.value.line,
		col:    s.value.col,
	})
}

func checkReferences(refs []reference, defined map[string]map[string]bool) error {
	for _, r := range refs {
		for _, name := range r.names {
			if !defined[r.target][name] {
				return newError(r.line, r.col, "unknown %s \"%s\" in %s", r.target, name, r.owner)
			}
		}
	}
	return nil
}

func unknownBlock(s statement, context string) *Error {
	return newError(s.line, s.col, "unknown block \"%s\" in %s", s.name, context)
}

func requires(block statement, what string) *Error {
	return newError(block.line, block.col, "%s \"%s\" requires %s", block.name, block.label, what)
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

func (a attributes) text(name string) string {
	return a[name].value.str
}

func (a attributes) list(name string) []string {
	if s, ok := a[name]; ok {
		return s.value.list
	}
	return []string{}
}

func (a attributes) number(name string) *int {
	s, ok := a[name]
	if !ok {
		return nil
	}
	n, err := strconv.Atoi(s.value.str)
	if err != nil {
		return nil
	}
	return &n
}

func buildFile(stmts []statement) (*Factoryfile, error) {
	file := &Factoryfile{Factories: []Factory{}}
	names := map[string]bool{}
	onBlock := func(s statement) error {
		if s.name != "factory" {
			return unknownBlock(s, fileSchema.context)
		}
		if names[s.label] {
			return newError(s.line, s.col, "duplicate factory \"%s\"", s.label)
		}
		names[s.label] = true
		factory, err := buildFactory(s)
		if err != nil {
			return err
		}
		file.Factories = append(file.Factories, factory)
		return nil
	}
	attrs, err := walk(fileSchema, statement{body: stmts}, nil, onBlock)
	if err != nil {
		return nil, err
	}
	if !attrs.has("version") {
		return nil, newError(1, 1, "missing required attribute \"version\"")
	}
	file.Version = attrs.text("version")
	return file, nil
}

func buildFactory(block statement) (Factory, error) {
	f := Factory{Name: block.label, Agents: []Agent{}, Stages: []Stage{}, Loops: []Loop{}}
	refs := []reference{}
	defined := map[string]map[string]bool{"agent": {}, "stage": {}, "loop": {}}
	onBlock := func(s statement) error {
		names, known := defined[s.name]
		if !known {
			return unknownBlock(s, factorySchema.context)
		}
		if names[s.label] {
			return newError(s.line, s.col, "duplicate %s \"%s\" in factory \"%s\"", s.name, s.label, block.label)
		}
		names[s.label] = true
		return f.add(s, &refs)
	}
	attrs, err := walk(factorySchema, block, &refs, onBlock)
	if err != nil {
		return Factory{}, err
	}
	if err := checkReferences(refs, defined); err != nil {
		return Factory{}, err
	}
	f.Description = attrs.optional("description")
	f.Entry = attrs.optional("entry")
	if f.Entry == nil && len(f.Stages) > 0 {
		first := f.Stages[0].Name
		f.Entry = &first
	}
	return f, nil
}

func (f *Factory) add(s statement, refs *[]reference) error {
	switch s.name {
	case "agent":
		item, err := buildAgent(s, refs)
		f.Agents = append(f.Agents, item)
		return err
	case "stage":
		item, err := buildStage(s, refs)
		f.Stages = append(f.Stages, item)
		return err
	default:
		item, err := buildLoop(s, refs)
		f.Loops = append(f.Loops, item)
		return err
	}
}

func buildAgent(block statement, refs *[]reference) (Agent, error) {
	attrs, err := walk(agentSchema, block, refs, nil)
	if err != nil {
		return Agent{}, err
	}
	hasSource, hasContent := attrs.has("source"), attrs.has("content")
	if !hasSource && !hasContent {
		return Agent{}, requires(block, "\"source\" or \"content\"")
	}
	if hasSource && hasContent {
		return Agent{}, newError(block.line, block.col, "agent \"%s\" cannot set both \"source\" and \"content\"", block.label)
	}
	return Agent{
		Name:        block.label,
		Description: attrs.optional("description"),
		Role:        attrs.optional("role"),
		Model:       attrs.optional("model"),
		Tools:       attrs.list("tools"),
		Source:      attrs.optional("source"),
		Content:     attrs.optional("content"),
	}, nil
}

func buildStage(block statement, refs *[]reference) (Stage, error) {
	attrs, err := walk(stageSchema, block, refs, nil)
	if err != nil {
		return Stage{}, err
	}
	if !attrs.has("agents") {
		return Stage{}, requires(block, "\"agents\"")
	}
	return Stage{
		Name:        block.label,
		Description: attrs.optional("description"),
		Agents:      attrs.list("agents"),
		Next:        attrs.list("next"),
		Artifacts:   attrs.list("artifacts"),
		Content:     attrs.optional("content"),
	}, nil
}

func buildLoop(block statement, refs *[]reference) (Loop, error) {
	attrs, err := walk(loopSchema, block, refs, nil)
	if err != nil {
		return Loop{}, err
	}
	for _, required := range []string{"from", "to"} {
		if !attrs.has(required) {
			return Loop{}, requires(block, "\""+required+"\"")
		}
	}
	return Loop{
		Name:        block.label,
		Description: attrs.optional("description"),
		From:        attrs.text("from"),
		To:          attrs.text("to"),
		Until:       attrs.optional("until"),
		Max:         attrs.number("max"),
	}, nil
}
