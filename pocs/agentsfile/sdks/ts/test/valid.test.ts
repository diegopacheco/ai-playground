import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { load } from "../src/index";

const dir = join(import.meta.dir, "../../../spec/fixtures/valid");

describe("full.agentsfile", () => {
  const file = load(join(dir, "full.agentsfile"));
  const [dev, ci] = file.profiles;

  test("profiles and items keep file order", () => {
    expect(file.version).toBe("1");
    expect(file.profiles.map((p) => p.name)).toEqual(["dev", "ci"]);
    expect(dev?.description).toBe("Everyday coding setup");
    expect(ci?.description).toBeUndefined();
    expect(dev?.skills.map((s) => s.name)).toEqual(["code-review", "release-notes"]);
    expect(ci?.mcps.map((m) => m.name)).toEqual(["docs", "events", "shared"]);
    expect(ci?.plugins.map((p) => p.name)).toEqual(["toolkit", "security"]);
    expect(dev?.plugins).toEqual([]);
  });

  test("inline skill carries its dedented body and extra files", () => {
    expect(dev?.skills).toStrictEqual([
      {
        name: "code-review",
        description: "Reviews the current diff",
        source: undefined,
        content: "# Code review\nRead the diff and list bugs first.\n\nThen list style issues.",
        files: [
          { path: "scripts/diff.sh", source: undefined, content: "#!/usr/bin/env bash\ngit diff --stat" },
          { path: "checklist.md", source: "./skills/code-review/checklist.md", content: undefined },
        ],
      },
      {
        name: "release-notes",
        description: undefined,
        source: "https://github.com/acme/skills/release-notes",
        content: undefined,
        files: [],
      },
    ]);
  });

  test("inline agent keeps the indentation beyond the common one", () => {
    expect(dev?.agents).toStrictEqual([
      {
        name: "tester",
        description: "Writes and runs tests",
        model: "claude-sonnet-5-5",
        tools: ["Read", "Grep", "Bash"],
        source: undefined,
        content: "You write tests first.\n  Keep each test small.",
      },
    ]);
  });

  test("commands are inline or referenced", () => {
    expect(dev?.commands).toStrictEqual([
      {
        name: "ship",
        description: "Runs tests and opens a PR",
        argumentHint: "<branch>",
        model: "claude-haiku-5-5",
        tools: ["Bash"],
        source: undefined,
        content: "Run the tests, then open a PR for $ARGUMENTS.",
      },
    ]);
    expect(ci?.commands).toStrictEqual([
      {
        name: "fix",
        description: undefined,
        argumentHint: undefined,
        model: undefined,
        tools: [],
        source: "./commands/fix.md",
        content: undefined,
      },
    ]);
  });

  test("stdio mcp keeps ${GITHUB_TOKEN} literal", () => {
    expect(dev?.mcps).toStrictEqual([
      {
        name: "github",
        description: undefined,
        source: undefined,
        transport: "stdio",
        command: "npx",
        args: ["-y", "@modelcontextprotocol/server-github"],
        env: { GITHUB_TOKEN: "${GITHUB_TOKEN}", LOG_LEVEL: "info" },
        url: undefined,
        headers: {},
      },
    ]);
  });

  test("http, sse and referenced mcps default args, env and headers", () => {
    expect(ci?.mcps).toStrictEqual([
      {
        name: "docs",
        description: undefined,
        source: undefined,
        transport: "http",
        command: undefined,
        args: [],
        env: {},
        url: "https://mcp.acme.dev/docs",
        headers: { Authorization: "Bearer ${DOCS_TOKEN}" },
      },
      {
        name: "events",
        description: undefined,
        source: undefined,
        transport: "sse",
        command: undefined,
        args: [],
        env: {},
        url: "https://mcp.acme.dev/events",
        headers: {},
      },
      {
        name: "shared",
        description: undefined,
        source: "./mcp/shared.json",
        transport: undefined,
        command: undefined,
        args: [],
        env: {},
        url: undefined,
        headers: {},
      },
    ]);
  });

  test("quoted strings decode escapes", () => {
    expect(ci?.agents).toStrictEqual([
      {
        name: "reviewer",
        description: 'Says "no" to\tsloppy code',
        model: undefined,
        tools: [],
        source: "./agents/reviewer.md",
        content: undefined,
      },
    ]);
  });

  test("plugins are inline bundles or referenced", () => {
    const [toolkit, security] = ci?.plugins ?? [];
    expect(toolkit?.version).toBe("1.2.0");
    expect(toolkit?.author).toBe("Platform team");
    expect(toolkit?.description).toBe("Shared team tooling");
    expect(toolkit?.source).toBeUndefined();
    expect(toolkit?.skills.map((s) => [s.name, s.content])).toEqual([["lint", "Run the linter and fix every error."]]);
    expect(toolkit?.agents.map((a) => [a.name, a.source])).toEqual([["fixer", "./agents/fixer.md"]]);
    expect(toolkit?.commands.map((c) => [c.name, c.source])).toEqual([["fix", "./commands/fix.md"]]);
    expect(toolkit?.mcps.map((m) => [m.name, m.transport, m.url])).toEqual([
      ["search", "http", "https://mcp.acme.dev/search"],
    ]);
    expect(security).toStrictEqual({
      name: "security",
      description: undefined,
      version: undefined,
      author: undefined,
      source: "https://github.com/acme/plugins/security",
      skills: [],
      agents: [],
      commands: [],
      mcps: [],
    });
  });
});

describe("heredoc.agentsfile", () => {
  const file = load(join(dir, "heredoc.agentsfile"));
  const edge = file.profiles[0];

  test("a dedented heredoc version is accepted", () => {
    expect(file.version).toBe("1");
  });

  test("an empty body is the empty string", () => {
    expect(edge?.description).toBe("");
  });

  test("tabs are common indentation and a trailing blank line stays", () => {
    expect(edge?.skills[0]?.content).toBe("first\n  second\n");
  });

  test("nothing inside a heredoc is special", () => {
    expect(edge?.skills[1]?.content).toBe('"quoted" \\n ${HOME} # not a comment');
  });
});

describe("minimal.agentsfile", () => {
  test("has a version and zero profiles", () => {
    const file = load(join(dir, "minimal.agentsfile"));
    expect(file).toStrictEqual({ version: "1", profiles: [] });
  });
});
