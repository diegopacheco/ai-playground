package agentsfile

import "fmt"

type Error struct {
	Line    int
	Col     int
	Message string
}

func (e *Error) Error() string {
	return fmt.Sprintf("line %d, col %d: %s", e.Line, e.Col, e.Message)
}

func newError(line, col int, format string, args ...any) *Error {
	return &Error{Line: line, Col: col, Message: fmt.Sprintf(format, args...)}
}
