package factoryfile

import "os"

func Parse(src string) (*Factoryfile, error) {
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

func ParseFile(path string) (*Factoryfile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(string(data))
}
