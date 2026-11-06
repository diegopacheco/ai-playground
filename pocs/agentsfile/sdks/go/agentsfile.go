package agentsfile

import "os"

func Parse(src string) (*Agentsfile, error) {
	tokens, err := tokenize(src)
	if err != nil {
		return nil, err
	}
	stmts, err := parse(tokens)
	if err != nil {
		return nil, err
	}
	return buildFile(stmts)
}

func ParseFile(path string) (*Agentsfile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(string(data))
}
