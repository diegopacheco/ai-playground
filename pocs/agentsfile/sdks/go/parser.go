package agentsfile

type valueKind int

const (
	valueString valueKind = iota
	valueList
	valueMap
)

var valueKindNames = map[valueKind]string{
	valueString: "a string",
	valueList:   "a list",
	valueMap:    "a map",
}

type value struct {
	kind    valueKind
	line    int
	col     int
	str     string
	list    []string
	entries map[string]string
}

type statement struct {
	name    string
	line    int
	col     int
	isBlock bool
	value   value
	label   string
	body    []statement
}

type parser struct {
	tokens []token
	pos    int
}

func parse(tokens []token) ([]statement, error) {
	p := &parser{tokens: tokens}
	return p.statements(true)
}

func (p *parser) peek() token {
	return p.tokens[p.pos]
}

func (p *parser) next() token {
	t := p.tokens[p.pos]
	if p.pos < len(p.tokens)-1 {
		p.pos++
	}
	return t
}

func unexpected(t token, expected string) *Error {
	return newError(t.line, t.col, "expected %s, found %s", expected, t.kind)
}

func (p *parser) statements(top bool) ([]statement, error) {
	stmts := []statement{}
	for {
		t := p.peek()
		if top && t.kind == tokEOF {
			return stmts, nil
		}
		if !top && t.kind == tokRBrace {
			p.next()
			return stmts, nil
		}
		if t.kind != tokIdent {
			if top {
				return nil, unexpected(t, "identifier")
			}
			return nil, unexpected(t, "identifier or '}'")
		}
		s, err := p.statement()
		if err != nil {
			return nil, err
		}
		stmts = append(stmts, s)
	}
}

func (p *parser) statement() (statement, error) {
	name := p.next()
	s := statement{name: name.text, line: name.line, col: name.col}
	t := p.next()
	switch t.kind {
	case tokEquals:
		v, err := p.value()
		if err != nil {
			return s, err
		}
		s.value = v
	case tokString:
		brace := p.next()
		if brace.kind != tokLBrace {
			return s, unexpected(brace, "'{'")
		}
		body, err := p.statements(false)
		if err != nil {
			return s, err
		}
		s.isBlock = true
		s.label = t.text
		s.body = body
	default:
		return s, unexpected(t, "'=' or block label")
	}
	return s, nil
}

func (p *parser) value() (value, error) {
	t := p.next()
	v := value{line: t.line, col: t.col}
	switch t.kind {
	case tokString:
		v.kind = valueString
		v.str = t.text
		return v, nil
	case tokLBracket:
		v.kind = valueList
		list, err := p.list()
		v.list = list
		return v, err
	case tokLBrace:
		v.kind = valueMap
		entries, err := p.mapEntries()
		v.entries = entries
		return v, err
	default:
		return v, unexpected(t, "value")
	}
}

func (p *parser) list() ([]string, error) {
	items := []string{}
	if p.peek().kind == tokRBracket {
		p.next()
		return items, nil
	}
	for {
		item := p.next()
		if item.kind != tokString {
			return nil, unexpected(item, "string")
		}
		items = append(items, item.text)
		t := p.next()
		switch t.kind {
		case tokRBracket:
			return items, nil
		case tokComma:
			if p.peek().kind == tokRBracket {
				p.next()
				return items, nil
			}
		default:
			return nil, unexpected(t, "',' or ']'")
		}
	}
}

func (p *parser) mapEntries() (map[string]string, error) {
	entries := map[string]string{}
	for {
		key := p.next()
		if key.kind == tokRBrace {
			return entries, nil
		}
		if key.kind != tokIdent {
			return nil, unexpected(key, "identifier or '}'")
		}
		if _, exists := entries[key.text]; exists {
			return nil, newError(key.line, key.col, "duplicate key \"%s\"", key.text)
		}
		if eq := p.next(); eq.kind != tokEquals {
			return nil, unexpected(eq, "'='")
		}
		v := p.next()
		if v.kind != tokString {
			return nil, unexpected(v, "string")
		}
		entries[key.text] = v.text
		if p.peek().kind == tokComma {
			p.next()
		}
	}
}
