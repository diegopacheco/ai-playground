package agentsfile

import "strings"

type tokenKind int

const (
	tokIdent tokenKind = iota
	tokString
	tokEquals
	tokLBrace
	tokRBrace
	tokLBracket
	tokRBracket
	tokComma
	tokEOF
)

var tokenNames = map[tokenKind]string{
	tokIdent:    "identifier",
	tokString:   "string",
	tokEquals:   "'='",
	tokLBrace:   "'{'",
	tokRBrace:   "'}'",
	tokLBracket: "'['",
	tokRBracket: "']'",
	tokComma:    "','",
	tokEOF:      "end of file",
}

func (k tokenKind) String() string {
	return tokenNames[k]
}

var punctuation = map[rune]tokenKind{
	'=': tokEquals,
	'{': tokLBrace,
	'}': tokRBrace,
	'[': tokLBracket,
	']': tokRBracket,
	',': tokComma,
}

var escapes = map[rune]rune{
	'"':  '"',
	'\\': '\\',
	'n':  '\n',
	't':  '\t',
}

type token struct {
	kind tokenKind
	text string
	line int
	col  int
}

type lexer struct {
	runes  []rune
	pos    int
	line   int
	col    int
	tokens []token
}

func tokenize(src string) ([]token, error) {
	l := &lexer{runes: []rune(src), line: 1, col: 1}
	for !l.done() {
		if err := l.scan(); err != nil {
			return nil, err
		}
	}
	l.emit(tokEOF, "", l.line, l.col)
	return l.tokens, nil
}

func (l *lexer) done() bool {
	return l.pos >= len(l.runes)
}

func (l *lexer) current() rune {
	return l.runes[l.pos]
}

func (l *lexer) advance() {
	if l.current() == '\n' {
		l.line++
		l.col = 1
	} else {
		l.col++
	}
	l.pos++
}

func (l *lexer) emit(kind tokenKind, text string, line, col int) {
	l.tokens = append(l.tokens, token{kind: kind, text: text, line: line, col: col})
}

func (l *lexer) scan() error {
	r := l.current()
	switch {
	case r == ' ' || r == '\t' || r == '\r' || r == '\n':
		l.advance()
	case r == '#':
		for !l.done() && l.current() != '\n' {
			l.advance()
		}
	case isIdentStart(r):
		l.scanIdent()
	case r == '"':
		return l.scanString()
	case r == '<' && l.peekAt(1) == '<':
		return l.scanHeredoc()
	default:
		kind, ok := punctuation[r]
		if !ok {
			return newError(l.line, l.col, "unexpected character '%c'", r)
		}
		l.emit(kind, string(r), l.line, l.col)
		l.advance()
	}
	return nil
}

func (l *lexer) scanIdent() {
	line, col, start := l.line, l.col, l.pos
	for !l.done() && isIdentPart(l.current()) {
		l.advance()
	}
	l.emit(tokIdent, string(l.runes[start:l.pos]), line, col)
}

func (l *lexer) scanString() error {
	line, col := l.line, l.col
	l.advance()
	var text strings.Builder
	for {
		if l.done() || l.current() == '\n' {
			return newError(line, col, "unterminated string")
		}
		r := l.current()
		switch r {
		case '"':
			l.advance()
			l.emit(tokString, text.String(), line, col)
			return nil
		case '\\':
			escLine, escCol := l.line, l.col
			l.advance()
			if l.done() || l.current() == '\n' {
				return newError(line, col, "unterminated string")
			}
			decoded, ok := escapes[l.current()]
			if !ok {
				return newError(escLine, escCol, "invalid escape '\\%c'", l.current())
			}
			text.WriteRune(decoded)
			l.advance()
		default:
			text.WriteRune(r)
			l.advance()
		}
	}
}

func (l *lexer) peekAt(offset int) rune {
	if l.pos+offset >= len(l.runes) {
		return 0
	}
	return l.runes[l.pos+offset]
}

func (l *lexer) scanHeredoc() error {
	line, col := l.line, l.col
	l.advance()
	l.advance()
	start := l.pos
	for !l.done() && isMarkerPart(l.current()) {
		l.advance()
	}
	marker := string(l.runes[start:l.pos])
	if marker == "" || !isIdentStart(rune(marker[0])) {
		return newError(line, col, "invalid heredoc marker")
	}
	for !l.done() && isHeredocSpace(l.current()) {
		l.advance()
	}
	if l.done() {
		return newError(line, col, "unterminated heredoc")
	}
	if l.current() != '\n' {
		return newError(line, col, "invalid heredoc marker")
	}
	l.advance()
	var lines []string
	for {
		if l.done() {
			return newError(line, col, "unterminated heredoc")
		}
		begin := l.pos
		for !l.done() && l.current() != '\n' {
			l.advance()
		}
		text := string(l.runes[begin:l.pos])
		if !l.done() {
			l.advance()
		}
		if strings.Trim(text, " \t\r") == marker {
			l.emit(tokString, dedent(lines), line, col)
			return nil
		}
		lines = append(lines, text)
	}
}

func dedent(lines []string) string {
	indent := -1
	for _, line := range lines {
		if isBlank(line) {
			continue
		}
		count := len(line) - len(strings.TrimLeft(line, " \t"))
		if indent < 0 || count < indent {
			indent = count
		}
	}
	out := make([]string, len(lines))
	for i, line := range lines {
		if !isBlank(line) {
			out[i] = strings.TrimRight(line[indent:], "\r")
		}
	}
	return strings.Join(out, "\n")
}

func isBlank(line string) bool {
	return strings.Trim(line, " \t\r") == ""
}

func isHeredocSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\r'
}

func isMarkerPart(r rune) bool {
	return r == '_' || (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
}

func isIdentStart(r rune) bool {
	return r == '_' || (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z')
}

func isIdentPart(r rune) bool {
	return isIdentStart(r) || r == '-' || (r >= '0' && r <= '9')
}
