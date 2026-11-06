import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { AgentsfileError, load, parse } from "../src/index";

const dir = join(import.meta.dir, "../../../spec/fixtures/invalid");

function failure(run: () => unknown): AgentsfileError {
  try {
    run();
  } catch (error) {
    if (error instanceof AgentsfileError) {
      return error;
    }
    throw error;
  }
  throw new Error("expected an AgentsfileError");
}

const fixtures = readdirSync(dir).filter((name) => name.endsWith(".agentsfile")).sort();

describe("invalid fixtures", () => {
  test("all 51 fixtures are present", () => {
    expect(fixtures.length).toBe(51);
  });

  for (const name of fixtures) {
    test(name, () => {
      const path = join(dir, name);
      const header = readFileSync(path, "utf8").split("\n")[0] ?? "";
      const message = header.replace(/^# expect: /, "");
      const error = failure(() => load(path));
      expect(error).toBeInstanceOf(AgentsfileError);
      expect(error).toBeInstanceOf(Error);
      expect(error.message).toBe(message);
      const [, line, col] = /^line (\d+), col (\d+): /.exec(message) ?? [];
      expect(error.line).toBe(Number(line));
      expect(error.col).toBe(Number(col));
    });
  }
});

describe("error positions", () => {
  test("columns count code points, not UTF-16 units", () => {
    const error = failure(() => parse('version = "1"\nprofile "😀" { @ }'));
    expect(error.message).toBe("line 2, col 15: unexpected character '@'");
  });

  test("a lexical error wins over an earlier syntax error", () => {
    const error = failure(() => parse('version = = "1"\n"\n'));
    expect(error.message).toBe("line 2, col 1: unterminated string");
  });

  test("a syntax error wins over an earlier schema error", () => {
    const error = failure(() => parse('nope = "x"\nversion "1"'));
    expect(error.message).toBe("line 2, col 12: expected '{', found end of file");
  });

  test("missing version is reported after every profile", () => {
    const error = failure(() => parse('profile "p" {\n  skill "s" {\n  }\n}'));
    expect(error.message).toBe('line 2, col 3: skill "s" requires "source" or "content"');
  });

  test("positions after a heredoc follow its closing line", () => {
    const error = failure(() => parse("version = <<V\n1\nV\n  @"));
    expect(error.message).toBe("line 4, col 3: unexpected character '@'");
  });

  test("a single '<' is still an unexpected character", () => {
    const error = failure(() => parse("version = <V"));
    expect(error.message).toBe("line 1, col 11: unexpected character '<'");
  });

  test("a mcp conflict is reported before its transport rule", () => {
    const error = failure(() => parse('version = "1"\nprofile "p" {\n  mcp "m" {\n    source = "a"\n    transport = "stdio"\n  }\n}'));
    expect(error.message).toBe('line 3, col 3: mcp "m" cannot set both "source" and "transport"');
  });

  test("the same name in different kinds is not a duplicate", () => {
    const file = parse(
      'version = "1"\nprofile "p" {\n  skill "x" { source = "a" }\n  command "x" { source = "b" }\n  plugin "x" { source = "c" }\n}',
    );
    expect(file.profiles[0]?.skills[0]?.source).toBe("a");
    expect(file.profiles[0]?.commands[0]?.source).toBe("b");
    expect(file.profiles[0]?.plugins[0]?.source).toBe("c");
  });
});
