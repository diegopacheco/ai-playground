package factoryfile

import (
	"regexp"
	"sort"
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
	attrs: map[string]valueKind{
		"title":       valueString,
		"description": valueString,
		"entry":       valueString,
		"flow":        valueString,
		"backlog":     valueString,
	},
	refs: map[string]string{"entry": "stage"},
}

var agentSchema = schema{
	context: "agent",
	attrs: map[string]valueKind{
		"description": valueString,
		"role":        valueString,
		"provider":    valueString,
		"model":       valueString,
		"tools":       valueList,
		"source":      valueString,
		"content":     valueString,
	},
}

var stageSchema = schema{
	context: "stage",
	attrs: map[string]valueKind{
		"label":       valueString,
		"description": valueString,
		"agents":      valueList,
		"run":         valueString,
		"script":      valueString,
		"next":        valueList,
		"artifacts":   valueList,
		"output":      valueString,
		"check":       valueString,
		"verdict":     valueMap,
		"source":      valueString,
		"content":     valueString,
		"skills":      valueList,
		"tasks":       valueString,
		"parallel":    valueString,
		"visits":      valueString,
		"end":         valueString,
		"at":          valueList,
	},
	refs:  map[string]string{"agents": "agent", "next": "stage", "tasks": "stage"},
	check: checkStage,
}

var onSchema = schema{
	context: "on",
	attrs: map[string]valueKind{
		"to":     valueString,
		"label":  valueString,
		"reason": valueString,
		"cycle":  valueString,
	},
	refs:  map[string]string{"to": "stage"},
	check: checkCycle,
}

var exhaustedSchema = schema{
	context: "exhausted",
	attrs: map[string]valueKind{
		"label":  valueString,
		"reason": valueString,
	},
}

var verdictKeys = map[string]bool{"field": true, "yes": true, "no": true, "list": true, "notes": true}

var atPattern = regexp.MustCompile(`^(0|[1-9][0-9]{0,2})$`)

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
	if s.name == "max" {
		return checkNumber(s)
	}
	return nil
}

func checkNumber(s statement) error {
	if !maxPattern.MatchString(s.value.str) {
		return newError(s.value.line, s.value.col, "invalid %s \"%s\", expected an integer from 1 to 999999", s.name, s.value.str)
	}
	return nil
}

func checkStage(s statement) error {
	switch s.name {
	case "parallel", "visits":
		return checkNumber(s)
	case "at":
		if len(s.value.list) != 2 || !atPattern.MatchString(s.value.list[0]) || !atPattern.MatchString(s.value.list[1]) {
			return newError(s.value.line, s.value.col, "invalid at, expected two integers from 0 to 999")
		}
	case "verdict":
		keys := make([]string, 0, len(s.value.entries))
		for key := range s.value.entries {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			if !verdictKeys[key] {
				return newError(s.value.line, s.value.col, "unknown key \"%s\" in verdict", key)
			}
		}
		if _, ok := s.value.entries["field"]; !ok {
			return newError(s.value.line, s.value.col, "verdict requires \"field\"")
		}
	}
	return nil
}

func checkCycle(s statement) error {
	if s.name == "cycle" && s.value.str != "true" && s.value.str != "false" {
		return newError(s.value.line, s.value.col, "invalid cycle \"%s\", expected \"true\" or \"false\"", s.value.str)
	}
	return nil
}

func walk(sc schema, block statement, refs *[]reference, onBlock func(statement) error) (attributes, error) {
	return walkAs(sc, block, sc.context+" \""+block.label+"\"", refs, onBlock)
}

func walkAs(sc schema, block statement, owner string, refs *[]reference, onBlock func(statement) error) (attributes, error) {
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
		collectReference(sc, owner, refs, s)
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

func collectReference(sc schema, owner string, refs *[]reference, s statement) {
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
		owner:  owner,
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

func (a attributes) entries(name string) map[string]string {
	if s, ok := a[name]; ok {
		return s.value.entries
	}
	return nil
}

func (a attributes) numbers(name string) []int {
	out := []int{}
	if s, ok := a[name]; ok {
		for _, item := range s.value.list {
			n, _ := strconv.Atoi(item)
			out = append(out, n)
		}
	}
	return out
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
	f.Title = attrs.optional("title")
	f.Description = attrs.optional("description")
	f.Entry = attrs.optional("entry")
	f.Flow = attrs.optional("flow")
	f.Backlog = attrs.optional("backlog")
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
		Provider:    attrs.optional("provider"),
		Model:       attrs.optional("model"),
		Tools:       attrs.list("tools"),
		Source:      attrs.optional("source"),
		Content:     attrs.optional("content"),
	}, nil
}

func buildStage(block statement, refs *[]reference) (Stage, error) {
	owner := "stage \"" + block.label + "\""
	routes := []Route{}
	outcomes := map[string]bool{}
	onBlock := func(s statement) error {
		if s.name != "on" {
			return unknownBlock(s, stageSchema.context)
		}
		if outcomes[s.label] {
			return newError(s.line, s.col, "duplicate on \"%s\" in stage \"%s\"", s.label, block.label)
		}
		outcomes[s.label] = true
		route, err := buildRoute(s, owner, refs)
		routes = append(routes, route)
		return err
	}
	attrs, err := walkAs(stageSchema, block, owner, refs, onBlock)
	if err != nil {
		return Stage{}, err
	}
	if !attrs.has("agents") && !attrs.has("run") {
		return Stage{}, requires(block, "\"agents\" or \"run\"")
	}
	if attrs.has("source") && attrs.has("content") {
		return Stage{}, newError(block.line, block.col, "stage \"%s\" cannot set both \"source\" and \"content\"", block.label)
	}
	if attrs.has("next") && outcomes["next"] {
		return Stage{}, newError(block.line, block.col, "stage \"%s\" cannot set both \"next\" and on \"next\"", block.label)
	}
	if attrs.has("next") || outcomes["next"] {
		for _, route := range routes {
			if route.Outcome != "next" {
				return Stage{}, newError(block.line, block.col, "stage \"%s\" cannot mix the next route with on \"%s\"", block.label, route.Outcome)
			}
		}
	}
	return Stage{
		Name:        block.label,
		Label:       attrs.optional("label"),
		Description: attrs.optional("description"),
		Agents:      attrs.list("agents"),
		Run:         attrs.optional("run"),
		Script:      attrs.optional("script"),
		Next:        attrs.list("next"),
		Artifacts:   attrs.list("artifacts"),
		Output:      attrs.optional("output"),
		Check:       attrs.optional("check"),
		Verdict:     attrs.entries("verdict"),
		Source:      attrs.optional("source"),
		Content:     attrs.optional("content"),
		Skills:      attrs.list("skills"),
		Tasks:       attrs.optional("tasks"),
		Parallel:    attrs.number("parallel"),
		Visits:      attrs.number("visits"),
		End:         attrs.optional("end"),
		At:          attrs.numbers("at"),
		Routes:      routes,
	}, nil
}

func buildRoute(block statement, owner string, refs *[]reference) (Route, error) {
	var exhausted *Exhausted
	onBlock := exhaustedBlock(block, owner, refs, &exhausted)
	attrs, err := walkAs(onSchema, block, owner, refs, onBlock)
	if err != nil {
		return Route{}, err
	}
	if !attrs.has("to") {
		return Route{}, requires(block, "\"to\"")
	}
	return Route{
		Outcome:   block.label,
		To:        attrs.text("to"),
		Label:     attrs.optional("label"),
		Reason:    attrs.optional("reason"),
		Cycle:     attrs.text("cycle") == "true",
		Exhausted: exhausted,
	}, nil
}

func exhaustedBlock(parent statement, owner string, refs *[]reference, into **Exhausted) func(statement) error {
	return func(s statement) error {
		if s.name != "exhausted" {
			return unknownBlock(s, parent.name)
		}
		if *into != nil {
			return newError(s.line, s.col, "duplicate exhausted in %s \"%s\"", parent.name, parent.label)
		}
		*refs = append(*refs, reference{target: "stage", names: []string{s.label}, owner: owner, line: s.line, col: s.col})
		item, err := buildExhausted(s, owner, refs)
		*into = &item
		return err
	}
}

func buildExhausted(block statement, owner string, refs *[]reference) (Exhausted, error) {
	var next *Exhausted
	attrs, err := walkAs(exhaustedSchema, block, owner, refs, exhaustedBlock(block, owner, refs, &next))
	if err != nil {
		return Exhausted{}, err
	}
	return Exhausted{
		To:        block.label,
		Label:     attrs.optional("label"),
		Reason:    attrs.optional("reason"),
		Exhausted: next,
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
