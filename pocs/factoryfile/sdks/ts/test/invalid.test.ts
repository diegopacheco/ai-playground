import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { FactoryfileError, load, parse } from "../src/index";

const dir = join(import.meta.dir, "../../../spec/fixtures/invalid");

function failure(run: () => unknown): FactoryfileError {
  try {
    run();
  } catch (error) {
    if (error instanceof FactoryfileError) {
      return error;
    }
    throw error;
  }
  throw new Error("expected a FactoryfileError");
}

const fixtures = readdirSync(dir).filter((name) => name.endsWith(".factoryfile")).sort();

describe("invalid fixtures", () => {
  test("all 63 fixtures are present", () => {
    expect(fixtures.length).toBe(63);
  });

  for (const name of fixtures) {
    test(name, () => {
      const path = join(dir, name);
      const header = readFileSync(path, "utf8").split("\n")[0] ?? "";
      const message = header.replace(/^# expect: /, "");
      const error = failure(() => load(path));
      expect(error).toBeInstanceOf(Error);
      expect(error.message).toBe(message);
      const [, line, col] = /^line (\d+), col (\d+): /.exec(message) ?? [];
      expect(error.line).toBe(Number(line));
      expect(error.col).toBe(Number(col));
    });
  }
});

describe("error positions", () => {
  test("columns count characters, not bytes", () => {
    const error = failure(() => parse('version = "1"\nfactory "é" { owner = "x" }\n'));
    expect(error.message).toBe('line 2, col 15: unknown attribute "owner" in factory');
  });

  test("columns count code points, not UTF-16 units", () => {
    const error = failure(() => parse('version = "1"\nfactory "😀" { @ }'));
    expect(error.message).toBe("line 2, col 15: unexpected character '@'");
  });

  test("references are checked after every statement of the factory", () => {
    const error = failure(() =>
      parse('version = "1"\nfactory "f" {\n  entry = "nope"\n  stage "s" { agents = [] }\n  agent "a" { }\n}'),
    );
    expect(error.message).toBe('line 5, col 3: agent "a" requires "source" or "content"');
  });

  test("references are checked in document order", () => {
    const error = failure(() =>
      parse('version = "1"\nfactory "f" {\n  stage "s" { agents = ["x"] next = ["y"] }\n  entry = "z"\n}'),
    );
    expect(error.message).toBe('line 3, col 24: unknown agent "x" in stage "s"');
  });

  test("missing version is reported after every factory", () => {
    const error = failure(() => parse('factory "f" {\n  loop "l" { }\n}'));
    expect(error.message).toBe('line 2, col 3: loop "l" requires "from"');
  });
});
