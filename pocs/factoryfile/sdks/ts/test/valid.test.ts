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

describe("routes.factoryfile", () => {
  const gate = load(join(dir, "routes.factoryfile")).factories[0];
  const scrum = load(join(dir, "full.factoryfile")).factories[0];

  test("a v1.1 factory carries its title, flow, backlog and agent providers", () => {
    expect(gate?.title).toBe("Approval Gate");
    expect(gate?.flow).toBe("pull");
    expect(gate?.backlog).toBe("optional");
    expect(agent(gate, "maker")?.provider).toBe("codex");
    expect(agent(gate, "approver")?.provider).toBe("claude");
    expect(scrum !== undefined && "title" in scrum).toBe(false);
  });

  test("command stages need no agents and keep their run and end", () => {
    const test = stage(gate, "test");
    expect(test?.run).toBe("test");
    expect(test?.agents).toEqual([]);
    expect(test?.label).toBe("Test");
    expect(stage(gate, "ship")?.script).toBe("scripts/ship.sh");
    expect(stage(gate, "done")?.end).toBe("succeeded");
    expect(stage(gate, "done")?.at).toEqual([1, 1]);
    expect(stage(scrum, "build")?.at).toEqual([]);
  });

  test("agent stages carry their prompt file, skills, output, check and verdict", () => {
    const implement = stage(gate, "implement");
    expect(implement?.source).toBe("prompts/implement.md");
    expect(implement?.skills).toEqual(["skills/tdd.md", "skills/git.md"]);
    expect(implement?.visits).toBe(4);
    const approve = stage(gate, "approve");
    expect(approve?.output).toBe("approval.json");
    expect(approve?.check).toBe("verdict");
    expect(approve?.verdict).toStrictEqual({
      field: "approved",
      yes: "approved",
      no: "rejected",
      list: "reasons",
      notes: "notes",
    });
    const build = stage(gate, "build");
    expect(build?.tasks).toBe("split");
    expect(build?.parallel).toBe(10);
    expect(build !== undefined && "verdict" in build).toBe(false);
  });

  test("routes branch by outcome and fall back when their target is exhausted", () => {
    expect(stage(gate, "test")?.routes).toStrictEqual([
      { outcome: "passed", to: "approve", label: "pass", cycle: false },
      { outcome: "failed", to: "implement", label: "fail", reason: "the tests failed", cycle: false },
    ]);
    expect(stage(gate, "approve")?.routes[1]).toStrictEqual({ outcome: "rejected", to: "implement", cycle: true });
    expect(stage(gate, "implement")?.routes).toStrictEqual([
      {
        outcome: "next",
        to: "test",
        cycle: false,
        exhausted: {
          to: "approve",
          label: "first fallback",
          reason: "test already ran as often as allowed",
          exhausted: { to: "done", label: "last fallback" },
        },
      },
    ]);
    expect(stage(gate, "split")?.routes).toEqual([]);
  });
});
