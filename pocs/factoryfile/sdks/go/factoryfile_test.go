package factoryfile_test

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	factoryfile "github.com/diegopacheco/factoryfile/sdks/go"
)

const fixtures = "../../spec/fixtures"

const invalidFixtures = 63

func TestInvalidFixturesReportTheExactSpecError(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join(fixtures, "invalid", "*.factoryfile"))
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) != invalidFixtures {
		t.Fatalf("expected %d invalid fixtures, found %d", invalidFixtures, len(paths))
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
			file, err := factoryfile.ParseFile(path)
			if err == nil {
				t.Fatalf("expected error %q, parsed %+v", expected, file)
			}
			if err.Error() != expected {
				t.Fatalf("got %q, want %q", err.Error(), expected)
			}
			var specErr *factoryfile.Error
			if !errors.As(err, &specErr) {
				t.Fatalf("error %T is not *factoryfile.Error", err)
			}
		})
	}
}

func parseValid(t *testing.T, name string) *factoryfile.Factoryfile {
	t.Helper()
	file, err := factoryfile.ParseFile(filepath.Join(fixtures, "valid", name))
	if err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	return file
}

func text(s string) *string {
	return &s
}

func number(n int) *int {
	return &n
}

func equal(t *testing.T, label string, got, want any) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("%s: got %#v, want %#v", label, got, want)
	}
}

func TestMinimalFileHasNoFactories(t *testing.T) {
	file := parseValid(t, "minimal.factoryfile")
	equal(t, "version", file.Version, "1")
	equal(t, "factories", file.Factories, []factoryfile.Factory{})
}

func TestFactoriesAndItemsKeepDocumentOrder(t *testing.T) {
	file := parseValid(t, "full.factoryfile")
	factories := []string{}
	for _, f := range file.Factories {
		factories = append(factories, f.Name)
	}
	equal(t, "factories", factories, []string{"scrum", "solo", "empty"})
	stages := []string{}
	for _, s := range file.Factories[0].Stages {
		stages = append(stages, s.Name)
	}
	equal(t, "stages", stages, []string{"refinement", "planning", "build", "review", "retro"})
	loops := []string{}
	for _, l := range file.Factories[0].Loops {
		loops = append(loops, l.Name)
	}
	equal(t, "loops", loops, []string{"sprint", "daily"})
}

func TestExplicitEntryWinsAndMissingEntryFallsBackToTheFirstStage(t *testing.T) {
	file := parseValid(t, "full.factoryfile")
	equal(t, "explicit entry", file.Factories[0].Entry, text("refinement"))
	equal(t, "first stage entry", file.Factories[1].Entry, text("code"))
	equal(t, "no stage entry", file.Factories[2].Entry, (*string)(nil))
}

func TestInlineAgentCarriesItsWholePromptAndSourceAgentPointsAway(t *testing.T) {
	scrum := parseValid(t, "full.factoryfile").Factories[0]
	po := scrum.Agent("product-owner")
	if po == nil {
		t.Fatal("product-owner agent is missing")
	}
	equal(t, "role", po.Role, text("product owner"))
	equal(t, "model", po.Model, text("claude-opus-5-5"))
	equal(t, "tools", po.Tools, []string{"Read", "Write"})
	equal(t, "source", po.Source, (*string)(nil))
	equal(t, "content", po.Content, text("You keep the backlog ordered by value.\n\nEvery story has acceptance criteria."))
	dev := scrum.Agent("developer")
	if dev == nil {
		t.Fatal("developer agent is missing")
	}
	equal(t, "developer source", dev.Source, text("./agents/developer.md"))
	equal(t, "developer content", dev.Content, (*string)(nil))
	equal(t, "developer model", dev.Model, (*string)(nil))
}

func TestStagesHoldTheGraphEdgesAndTheirArtifacts(t *testing.T) {
	scrum := parseValid(t, "full.factoryfile").Factories[0]
	build := scrum.Stage("build")
	if build == nil {
		t.Fatal("build stage is missing")
	}
	equal(t, "build agents", build.Agents, []string{"developer"})
	equal(t, "build next", build.Next, []string{"review", "retro"})
	equal(t, "build artifacts", build.Artifacts, []string{"src/", "tests/"})
	equal(t, "a stage without next ends the factory", scrum.Stage("review").Next, []string{})
	equal(t, "retro agents", scrum.Stage("retro").Agents, []string{})
	equal(t, "planning content", scrum.Stage("planning").Content, text("Pick the stories for the sprint goal."))
}

func TestLoopsAreBackEdgesWithAnExitConditionAndABound(t *testing.T) {
	loops := parseValid(t, "full.factoryfile").Factories[0].Loops
	sprint := loops[0]
	equal(t, "sprint edge", []string{sprint.From, sprint.To}, []string{"review", "planning"})
	equal(t, "sprint until", sprint.Until, text("the backlog is empty"))
	equal(t, "sprint max", sprint.Max, number(26))
	daily := loops[1]
	equal(t, "daily edge", []string{daily.From, daily.To}, []string{"build", "build"})
	equal(t, "daily until", daily.Until, (*string)(nil))
	equal(t, "daily max", daily.Max, (*int)(nil))
}

func TestHeredocsStripSharedIndentAndKeepTextRaw(t *testing.T) {
	file := parseValid(t, "heredoc.factoryfile")
	equal(t, "version", file.Version, "1")
	edge := file.Factories[0]
	equal(t, "description", edge.Description, text(""))
	equal(t, "first agent", edge.Agents[0].Content, text("first\n  second\n"))
	equal(t, "raw agent", edge.Agents[1].Content, text("\"quoted\" \\n ${HOME} # not a comment"))
	equal(t, "stage agents", edge.Stages[0].Agents, []string{"tabs", "raw"})
}

func TestColumnsCountCharactersNotBytes(t *testing.T) {
	_, err := factoryfile.Parse("version = \"1\"\nfactory \"é\" { owner = \"x\" }\n")
	if err == nil {
		t.Fatal("expected an error")
	}
	equal(t, "error", err.Error(), "line 2, col 15: unknown attribute \"owner\" in factory")
}

func TestACycleThroughNextIsLegalForAParser(t *testing.T) {
	src := "version = \"1\"\nfactory \"x\" {\n  stage \"a\" { agents = [] next = [\"b\"] }\n  stage \"b\" { agents = [] next = [\"a\"] }\n}\n"
	file, err := factoryfile.Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "next", file.Factories[0].Stages[1].Next, []string{"a"})
}
