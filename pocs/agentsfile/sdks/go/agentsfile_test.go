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
	if len(paths) == 0 {
		t.Fatal("no invalid fixtures found")
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

func TestMinimalHasVersionAndNoProfiles(t *testing.T) {
	file := parseValid(t, "minimal.agentsfile")
	if file.Version != "1" {
		t.Fatalf("version %q", file.Version)
	}
	if file.Profiles == nil || len(file.Profiles) != 0 {
		t.Fatalf("profiles must be empty and non-nil, got %#v", file.Profiles)
	}
}

func TestFullKeepsProfilesInFileOrder(t *testing.T) {
	file := parseValid(t, "full.agentsfile")
	if file.Version != "1" {
		t.Fatalf("version %q", file.Version)
	}
	var names []string
	for _, p := range file.Profiles {
		names = append(names, p.Name)
	}
	if !reflect.DeepEqual(names, []string{"dev", "ci"}) {
		t.Fatalf("profiles %v", names)
	}
}

func TestFullDevProfile(t *testing.T) {
	dev := parseValid(t, "full.agentsfile").Profiles[0]
	if !reflect.DeepEqual(dev.Description, text("Everyday coding setup")) {
		t.Fatalf("description %v", dev.Description)
	}
	wantSkills := []agentsfile.Skill{
		{Name: "code-review", Source: "./skills/code-review", Description: text("Reviews the current diff")},
		{Name: "release-notes", Source: "https://github.com/acme/skills/release-notes"},
	}
	if !reflect.DeepEqual(dev.Skills, wantSkills) {
		t.Fatalf("skills %+v", dev.Skills)
	}
	wantAgents := []agentsfile.Agent{
		{Name: "tester", Source: "./agents/tester.md", Model: text("claude-sonnet-5-5"), Tools: []string{"Read", "Grep", "Bash"}},
	}
	if !reflect.DeepEqual(dev.Agents, wantAgents) {
		t.Fatalf("agents %+v", dev.Agents)
	}
	wantCommands := []agentsfile.Command{
		{Name: "ship", Source: "./commands/ship.md", Description: text("Runs tests and opens a PR")},
	}
	if !reflect.DeepEqual(dev.Commands, wantCommands) {
		t.Fatalf("commands %+v", dev.Commands)
	}
	wantMcps := []agentsfile.Mcp{
		{
			Name:      "github",
			Transport: agentsfile.Stdio,
			Command:   text("npx"),
			Args:      []string{"-y", "@modelcontextprotocol/server-github"},
			Env:       map[string]string{"GITHUB_TOKEN": "${GITHUB_TOKEN}", "LOG_LEVEL": "info"},
			Headers:   map[string]string{},
		},
	}
	if !reflect.DeepEqual(dev.Mcps, wantMcps) {
		t.Fatalf("mcps %+v", dev.Mcps)
	}
}

func TestFullCiProfile(t *testing.T) {
	ci := parseValid(t, "full.agentsfile").Profiles[1]
	if ci.Description != nil {
		t.Fatalf("description must be absent, got %q", *ci.Description)
	}
	if ci.Skills == nil || len(ci.Skills) != 0 || ci.Commands == nil || len(ci.Commands) != 0 {
		t.Fatalf("skills and commands must be empty and non-nil: %#v %#v", ci.Skills, ci.Commands)
	}
	wantAgents := []agentsfile.Agent{
		{Name: "reviewer", Source: "./agents/reviewer.md", Description: text("Says \"no\" to\tsloppy code"), Tools: []string{}},
	}
	if !reflect.DeepEqual(ci.Agents, wantAgents) {
		t.Fatalf("agents %+v", ci.Agents)
	}
	wantMcps := []agentsfile.Mcp{
		{
			Name:      "docs",
			Transport: agentsfile.HTTP,
			URL:       text("https://mcp.acme.dev/docs"),
			Args:      []string{},
			Env:       map[string]string{},
			Headers:   map[string]string{"Authorization": "Bearer ${DOCS_TOKEN}"},
		},
		{
			Name:      "events",
			Transport: agentsfile.SSE,
			URL:       text("https://mcp.acme.dev/events"),
			Args:      []string{},
			Env:       map[string]string{},
			Headers:   map[string]string{},
		},
	}
	if !reflect.DeepEqual(ci.Mcps, wantMcps) {
		t.Fatalf("mcps %+v", ci.Mcps)
	}
}

func TestDefaultsAreEmptyButNeverNil(t *testing.T) {
	ci := parseValid(t, "full.agentsfile").Profiles[1]
	if ci.Agents[0].Tools == nil {
		t.Fatal("missing tools must be an empty non-nil list")
	}
	events := ci.Mcps[1]
	if events.Args == nil || events.Env == nil || events.Headers == nil {
		t.Fatalf("missing args, env and headers must be non-nil: %#v", events)
	}
	if events.Command != nil {
		t.Fatalf("command must be absent for sse, got %q", *events.Command)
	}
}

func TestColumnsCountCharactersNotBytes(t *testing.T) {
	_, err := agentsfile.Parse("version = \"héllo wörld\" @")
	if err == nil || err.Error() != "line 1, col 25: unexpected character '@'" {
		t.Fatalf("got %v", err)
	}
}

func TestLexErrorWinsOverEarlierSchemaError(t *testing.T) {
	_, err := agentsfile.Parse("name = \"x\"\nversion = \"1\" $")
	if err == nil || err.Error() != "line 2, col 15: unexpected character '$'" {
		t.Fatalf("got %v", err)
	}
}

func TestMissingVersionIsReportedAfterProfiles(t *testing.T) {
	_, err := agentsfile.Parse("profile \"dev\" {\n  skill \"a\" {}\n}")
	if err == nil || err.Error() != "line 2, col 3: missing required attribute \"source\" in skill \"a\"" {
		t.Fatalf("got %v", err)
	}
}

func TestParseFileReportsMissingFile(t *testing.T) {
	_, err := agentsfile.ParseFile(filepath.Join(t.TempDir(), "Agentsfile"))
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("got %v", err)
	}
}
