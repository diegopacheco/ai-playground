import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { load } from "../src/index";

const dir = join(import.meta.dir, "../../../spec/fixtures/valid");

describe("full.agentsfile", () => {
  const file = load(join(dir, "full.agentsfile"));
  const [dev, ci] = file.profiles;

  test("version and profile order", () => {
    expect(file.version).toBe("1");
    expect(file.profiles.map((p) => p.name)).toEqual(["dev", "ci"]);
    expect(dev?.description).toBe("Everyday coding setup");
    expect(ci?.description).toBeUndefined();
  });

  test("dev skills keep order and optional description", () => {
    expect(dev?.skills).toStrictEqual([
      { name: "code-review", source: "./skills/code-review", description: "Reviews the current diff" },
      { name: "release-notes", source: "https://github.com/acme/skills/release-notes", description: undefined },
    ]);
    expect(ci?.skills).toEqual([]);
  });

  test("dev agent accepts a trailing comma in tools", () => {
    expect(dev?.agents).toStrictEqual([
      {
        name: "tester",
        source: "./agents/tester.md",
        description: undefined,
        model: "claude-sonnet-5-5",
        tools: ["Read", "Grep", "Bash"],
      },
    ]);
  });

  test("dev command", () => {
    expect(dev?.commands).toStrictEqual([
      { name: "ship", source: "./commands/ship.md", description: "Runs tests and opens a PR" },
    ]);
    expect(ci?.commands).toEqual([]);
  });

  test("stdio mcp keeps ${GITHUB_TOKEN} literal and defaults headers", () => {
    expect(dev?.mcps).toStrictEqual([
      {
        name: "github",
        transport: "stdio",
        description: undefined,
        command: "npx",
        args: ["-y", "@modelcontextprotocol/server-github"],
        env: { GITHUB_TOKEN: "${GITHUB_TOKEN}", LOG_LEVEL: "info" },
        url: undefined,
        headers: {},
      },
    ]);
    expect(Object.keys(dev?.mcps[0]?.env ?? {})).toEqual(["GITHUB_TOKEN", "LOG_LEVEL"]);
  });

  test("ci agent decodes escapes and defaults tools", () => {
    expect(ci?.agents).toStrictEqual([
      {
        name: "reviewer",
        source: "./agents/reviewer.md",
        description: 'Says "no" to\tsloppy code',
        model: undefined,
        tools: [],
      },
    ]);
  });

  test("http and sse mcps default args, env and headers", () => {
    expect(ci?.mcps).toStrictEqual([
      {
        name: "docs",
        transport: "http",
        description: undefined,
        command: undefined,
        args: [],
        env: {},
        url: "https://mcp.acme.dev/docs",
        headers: { Authorization: "Bearer ${DOCS_TOKEN}" },
      },
      {
        name: "events",
        transport: "sse",
        description: undefined,
        command: undefined,
        args: [],
        env: {},
        url: "https://mcp.acme.dev/events",
        headers: {},
      },
    ]);
  });
});

describe("minimal.agentsfile", () => {
  test("has a version and zero profiles", () => {
    const file = load(join(dir, "minimal.agentsfile"));
    expect(file).toStrictEqual({ version: "1", profiles: [] });
  });
});
