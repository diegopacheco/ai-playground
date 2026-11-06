import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { load, parse, type Factory } from "../src/index";

const dir = join(import.meta.dir, "../../../spec/fixtures/valid");

const stage = (factory: Factory | undefined, name: string) => factory?.stages.find((s) => s.name === name);
const agent = (factory: Factory | undefined, name: string) => factory?.agents.find((a) => a.name === name);

describe("minimal.factoryfile", () => {
  test("a file with only a version has no factories", () => {
    const file = load(join(dir, "minimal.factoryfile"));
    expect(file.version).toBe("1");
    expect(file.factories).toEqual([]);
  });
});

describe("full.factoryfile", () => {
  const file = load(join(dir, "full.factoryfile"));
  const [scrum, solo, empty] = file.factories;

  test("factories and items keep document order", () => {
    expect(file.factories.map((f) => f.name)).toEqual(["scrum", "solo", "empty"]);
    expect(scrum?.stages.map((s) => s.name)).toEqual(["refinement", "planning", "build", "review", "retro"]);
    expect(scrum?.loops.map((l) => l.name)).toEqual(["sprint", "daily"]);
  });

  test("explicit entry wins and a missing entry falls back to the first stage", () => {
    expect(scrum?.entry).toBe("refinement");
    expect(solo?.entry).toBe("code");
    expect(empty).toBeDefined();
    expect(empty !== undefined && "entry" in empty).toBe(false);
  });

  test("an inline agent carries its whole prompt and a source agent points away", () => {
    expect(agent(scrum, "product-owner")).toStrictEqual({
      name: "product-owner",
      description: "Owns the backlog",
      role: "product owner",
      model: "claude-opus-5-5",
      tools: ["Read", "Write"],
      content: "You keep the backlog ordered by value.\n\nEvery story has acceptance criteria.",
    });
    expect(agent(scrum, "developer")).toStrictEqual({
      name: "developer",
      role: "developer",
      tools: ["Read", "Edit", "Bash"],
      source: "./agents/developer.md",
    });
  });

  test("stages hold the graph edges and their artifacts", () => {
    const build = stage(scrum, "build");
    expect(build?.agents).toEqual(["developer"]);
    expect(build?.next).toEqual(["review", "retro"]);
    expect(build?.artifacts).toEqual(["src/", "tests/"]);
    expect(stage(scrum, "review")?.next).toEqual([]);
    expect(stage(scrum, "retro")?.agents).toEqual([]);
    expect(stage(scrum, "planning")?.content).toBe("Pick the stories for the sprint goal.");
  });

  test("loops are back edges with an exit condition and an integer bound", () => {
    expect(scrum?.loops[0]).toStrictEqual({
      name: "sprint",
      description: "One sprint per pass",
      from: "review",
      to: "planning",
      until: "the backlog is empty",
      max: 26,
    });
    expect(scrum?.loops[1]).toStrictEqual({ name: "daily", from: "build", to: "build" });
  });
});

describe("heredoc.factoryfile", () => {
  test("heredocs strip shared indent and keep text raw", () => {
    const file = load(join(dir, "heredoc.factoryfile"));
    const edge = file.factories[0];
    expect(file.version).toBe("1");
    expect(edge?.description).toBe("");
    expect(edge?.agents[0]?.content).toBe("first\n  second\n");
    expect(edge?.agents[1]?.content).toBe('"quoted" \\n ${HOME} # not a comment');
    expect(edge?.stages[0]?.agents).toEqual(["tabs", "raw"]);
  });
});

describe("graph rules", () => {
  test("a cycle through next is legal for a parser", () => {
    const file = parse(
      'version = "1"\nfactory "x" {\n  stage "a" { agents = [] next = ["b"] }\n  stage "b" { agents = [] next = ["a"] }\n}\n',
    );
    expect(file.factories[0]?.stages[1]?.next).toEqual(["a"]);
  });

  test("the same name may be used by different kinds", () => {
    const file = parse(
      'version = "1"\nfactory "x" {\n  agent "a" { content = "c" }\n  stage "a" { agents = ["a"] }\n  loop "a" { from = "a" to = "a" }\n}\n',
    );
    expect(file.factories[0]?.entry).toBe("a");
    expect(file.factories[0]?.loops[0]?.from).toBe("a");
  });
});
