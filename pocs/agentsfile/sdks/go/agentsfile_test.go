package agentsfile_test

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	agentsfile "github.com/diegopacheco/agentsfile/sdks/go"
)

const fixtures = "../../spec/fixtures"

func TestInvalidFixturesReportTheExactSpecError(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join(fixtures, "invalid", "*.agentsfile"))
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) != 51 {
		t.Fatalf("expected 51 invalid fixtures, found %d", len(paths))
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			first, _, _ := strings.Cut(string(data), "\n")
			expected, ok := strings.CutPrefix(first, "# expect: ")
			if !ok {
				t.Fatalf("fixture has no expect line: %q", first)
			}
			file, err := agentsfile.ParseFile(path)
			if err == nil {
				t.Fatalf("expected error %q, parsed %+v", expected, file)
			}
			if err.Error() != expected {
				t.Fatalf("got %q, want %q", err.Error(), expected)
			}
			var specErr *agentsfile.Error
			if !errors.As(err, &specErr) {
				t.Fatalf("error %T is not *agentsfile.Error", err)
			}
		})
	}
}

func parseValid(t *testing.T, name string) *agentsfile.Agentsfile {
	t.Helper()
	file, err := agentsfile.ParseFile(filepath.Join(fixtures, "valid", name))
	if err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	return file
}

func text(s string) *string {
	return &s
}

func transport(t agentsfile.Transport) *agentsfile.Transport {
	return &t
}

func equal(t *testing.T, label string, got, want any) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("%s: got %#v, want %#v", label, got, want)
	}
}

func TestMinimalHasVersionAndNoProfiles(t *testing.T) {
	file := parseValid(t, "minimal.agentsfile")
	equal(t, "version", file.Version, "1")
	equal(t, "profiles", file.Profiles, []agentsfile.Profile{})
}

func TestFullKeepsProfileAndItemOrder(t *testing.T) {
	file := parseValid(t, "full.agentsfile")
	equal(t, "version", file.Version, "1")
	var profiles, skills, mcps, plugins []string
	for _, p := range file.Profiles {
		profiles = append(profiles, p.Name)
	}
	for _, s := range file.Profiles[0].Skills {
		skills = append(skills, s.Name)
	}
	for _, m := range file.Profiles[1].Mcps {
		mcps = append(mcps, m.Name)
	}
	for _, p := range file.Profiles[1].Plugins {
		plugins = append(plugins, p.Name)
	}
	equal(t, "profiles", profiles, []string{"dev", "ci"})
	equal(t, "skills", skills, []string{"code-review", "release-notes"})
	equal(t, "mcps", mcps, []string{"docs", "events", "shared"})
	equal(t, "plugins", plugins, []string{"toolkit", "security"})
}

func TestInlineSkillCarriesItsWholeBodyAndExtraFiles(t *testing.T) {
	dev := parseValid(t, "full.agentsfile").Profiles[0]
	equal(t, "skills", dev.Skills, []agentsfile.Skill{
		{
			Name:        "code-review",
			Description: text("Reviews the current diff"),
			Content:     text("# Code review\nRead the diff and list bugs first.\n\nThen list style issues."),
			Files: []agentsfile.SkillFile{
				{Path: "scripts/diff.sh", Content: text("#!/usr/bin/env bash\ngit diff --stat")},
				{Path: "checklist.md", Source: text("./skills/code-review/checklist.md")},
			},
		},
		{
			Name:   "release-notes",
			Source: text("https://github.com/acme/skills/release-notes"),
			Files:  []agentsfile.SkillFile{},
		},
	})
}

func TestInlineAgentAndCommandKeepTheirPrompts(t *testing.T) {
	file := parseValid(t, "full.agentsfile")
	dev, ci := file.Profiles[0], file.Profiles[1]
	equal(t, "dev agents", dev.Agents, []agentsfile.Agent{{
		Name:        "tester",
		Description: text("Writes and runs tests"),
		Model:       text("claude-sonnet-5-5"),
		Tools:       []string{"Read", "Grep", "Bash"},
		Content:     text("You write tests first.\n  Keep each test small."),
	}})
	equal(t, "dev commands", dev.Commands, []agentsfile.Command{{
		Name:         "ship",
		Description:  text("Runs tests and opens a PR"),
		ArgumentHint: text("<branch>"),
		Model:        text("claude-haiku-5-5"),
		Tools:        []string{"Bash"},
		Content:      text("Run the tests, then open a PR for $ARGUMENTS."),
	}})
	equal(t, "ci commands", ci.Commands, []agentsfile.Command{{
		Name:   "fix",
		Source: text("./commands/fix.md"),
		Tools:  []string{},
	}})
}

func TestMcpServersAreInlineOrReferenced(t *testing.T) {
	file := parseValid(t, "full.agentsfile")
	equal(t, "dev mcps", file.Profiles[0].Mcps, []agentsfile.Mcp{{
		Name:      "github",
		Transport: transport(agentsfile.Stdio),
		Command:   text("npx"),
		Args:      []string{"-y", "@modelcontextprotocol/server-github"},
		Env:       map[string]string{"GITHUB_TOKEN": "${GITHUB_TOKEN}", "LOG_LEVEL": "info"},
		Headers:   map[string]string{},
	}})
	equal(t, "ci mcps", file.Profiles[1].Mcps, []agentsfile.Mcp{
		{
			Name:      "docs",
			Transport: transport(agentsfile.HTTP),
			URL:       text("https://mcp.acme.dev/docs"),
			Args:      []string{},
			Env:       map[string]string{},
			Headers:   map[string]string{"Authorization": "Bearer ${DOCS_TOKEN}"},
		},
		{
			Name:      "events",
			Transport: transport(agentsfile.SSE),
			URL:       text("https://mcp.acme.dev/events"),
			Args:      []string{},
			Env:       map[string]string{},
			Headers:   map[string]string{},
		},
		{
			Name:    "shared",
			Source:  text("./mcp/shared.json"),
			Args:    []string{},
			Env:     map[string]string{},
			Headers: map[string]string{},
		},
	})
}

func TestPluginsAreInlineBundlesOrReferenced(t *testing.T) {
	ci := parseValid(t, "full.agentsfile").Profiles[1]
	equal(t, "plugins", ci.Plugins, []agentsfile.Plugin{
		{
			Name:        "toolkit",
			Description: text("Shared team tooling"),
			Version:     text("1.2.0"),
			Author:      text("Platform team"),
			Skills: []agentsfile.Skill{{
				Name:    "lint",
				Content: text("Run the linter and fix every error."),
				Files:   []agentsfile.SkillFile{},
			}},
			Agents:   []agentsfile.Agent{{Name: "fixer", Source: text("./agents/fixer.md"), Tools: []string{}}},
			Commands: []agentsfile.Command{{Name: "fix", Source: text("./commands/fix.md"), Tools: []string{}}},
			Mcps: []agentsfile.Mcp{{
				Name:      "search",
				Transport: transport(agentsfile.HTTP),
				URL:       text("https://mcp.acme.dev/search"),
				Args:      []string{},
				Env:       map[string]string{},
				Headers:   map[string]string{},
			}},
		},
		{
			Name:     "security",
			Source:   text("https://github.com/acme/plugins/security"),
			Skills:   []agentsfile.Skill{},
			Agents:   []agentsfile.Agent{},
			Commands: []agentsfile.Command{},
			Mcps:     []agentsfile.Mcp{},
		},
	})
}

func TestQuotedStringsDecodeEscapesAndOptionalFieldsDefault(t *testing.T) {
	file := parseValid(t, "full.agentsfile")
	ci := file.Profiles[1]
	equal(t, "ci description", ci.Description, (*string)(nil))
	equal(t, "ci skills", ci.Skills, []agentsfile.Skill{})
	equal(t, "dev plugins", file.Profiles[0].Plugins, []agentsfile.Plugin{})
	equal(t, "ci agents", ci.Agents, []agentsfile.Agent{{
		Name:        "reviewer",
		Source:      text("./agents/reviewer.md"),
		Description: text("Says \"no\" to\tsloppy code"),
		Tools:       []string{},
	}})
}

func TestHeredocEdgeCases(t *testing.T) {
	file := parseValid(t, "heredoc.agentsfile")
	equal(t, "version", file.Version, "1")
	edge := file.Profiles[0]
	equal(t, "description", edge.Description, text(""))
	equal(t, "tabs", edge.Skills[0].Content, text("first\n  second\n"))
	equal(t, "raw", edge.Skills[1].Content, text("\"quoted\" \\n ${HOME} # not a comment"))
}

func TestErrors(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want string
	}{
		{"columns count characters not bytes", "version = \"1\"\nprofile \"é\" { @ }", "line 2, col 15: unexpected character '@'"},
		{"positions after a heredoc follow its closing line", "version = <<V\n1\nV\n  @", "line 4, col 3: unexpected character '@'"},
		{"lex error wins over an earlier schema error", "name = \"x\"\nversion = \"1\" $", "line 2, col 15: unexpected character '$'"},
		{"missing version is checked after profiles", "profile \"dev\" {\n  skill \"a\" {}\n}", "line 2, col 3: skill \"a\" requires \"source\" or \"content\""},
		{"single angle bracket is not a heredoc", "version = <V", "line 1, col 11: unexpected character '<'"},
		{"heredoc without a closing marker", "version = <<V\n1\n", "line 1, col 11: unterminated heredoc"},
		{"conflict is checked before the transport rule", "version = \"1\"\nprofile \"p\" {\n mcp \"m\" {\n  source = \"a\"\n  transport = \"stdio\"\n }\n}", "line 3, col 2: mcp \"m\" cannot set both \"source\" and \"transport\""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := agentsfile.Parse(c.src)
			if err == nil || err.Error() != c.want {
				t.Fatalf("got %v, want %q", err, c.want)
			}
		})
	}
}

func TestSameNameInDifferentKindsIsAllowed(t *testing.T) {
	src := "version = \"1\"\nprofile \"p\" {\n skill \"x\" { source = \"a\" }\n agent \"x\" { source = \"b\" }\n plugin \"x\" { source = \"c\" }\n}"
	file, err := agentsfile.Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	p := file.Profiles[0]
	equal(t, "names", []string{p.Skills[0].Name, p.Agents[0].Name, p.Plugins[0].Name}, []string{"x", "x", "x"})
}

func TestParseFileReportsMissingFile(t *testing.T) {
	_, err := agentsfile.ParseFile(filepath.Join(t.TempDir(), "Agentsfile"))
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("got %v", err)
	}
}
