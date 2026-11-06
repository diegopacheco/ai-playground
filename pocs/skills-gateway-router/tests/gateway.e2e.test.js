import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readdirSync, existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const SKILL_DIR = join(ROOT, ".claude/skills/skills-gateway");

function gateway(args, env = {}) {
  const result = spawnSync("node", [join(ROOT, "gateway.js"), ...args], { cwd: "/", encoding: "utf8", env: { ...process.env, ...env } });
  return { code: result.status, out: result.stdout, err: result.stderr };
}

test("the gateway is the only skill the client registers", () => {
  assert.deepEqual(readdirSync(join(ROOT, ".claude/skills")), ["skills-gateway"]);
});

test("the command in the gateway SKILL.md resolves to gateway.js", () => {
  const command = readFileSync(join(SKILL_DIR, "SKILL.md"), "utf8").match(/node \$\{CLAUDE_SKILL_DIR\}(\S+)/)[1];
  assert.ok(existsSync(join(SKILL_DIR, command)));
});

test("list shows every upstream skill namespaced by its source, from any directory", () => {
  const { code, out } = gateway(["list"]);
  assert.equal(code, 0);
  for (const name of ["text__word-count", "text__slugify", "dev__json-format", "dev__commit-message"]) assert.match(out, new RegExp(`- ${name}: `));
});

test("each upstream skill is the top route for a request in its domain", () => {
  const cases = {
    "count the words in notes.txt": "text__word-count",
    "make a url slug for this blog title": "text__slugify",
    "pretty print and validate config.json": "dev__json-format",
    "write a commit message for the staged changes": "dev__commit-message",
  };
  for (const [request, expected] of Object.entries(cases)) {
    const { out } = gateway(["route", request]);
    assert.equal(out.split("\n")[0].split(":")[0], `- ${expected}`, request);
  }
});

test("load returns the upstream instructions with a runnable script path and the arguments", () => {
  const { code, out } = gateway(["load", "text__word-count", "hello", "world"]);
  assert.equal(code, 0);
  assert.ok(out.includes(`node ${join(ROOT, "upstream-skills/text-skills/word-count")}/word-count.js`));
  assert.ok(out.includes("Request: hello world"));
});

test("errors exit with distinct codes so the model can tell a typo from an outage", () => {
  assert.equal(gateway(["load", "text__nope"]).code, 2);
  assert.equal(gateway(["bogus"]).code, 2);
  const down = gateway(["load", "gone__x"], { SKILL_SOURCES: "gone=/nowhere/at/all" });
  assert.equal(down.code, 3);
  assert.match(down.err, /unreachable/);
});

test("SKILL_SOURCES overrides sources.env and a dead source only warns on list", () => {
  const { code, out, err } = gateway(["list"], { SKILL_SOURCES: `text=${join(ROOT, "upstream-skills/text-skills")},gone=/nowhere/at/all` });
  assert.equal(code, 0);
  assert.match(out, /text__slugify/);
  assert.doesNotMatch(out, /dev__/);
  assert.match(err, /warning: skill source unreachable: gone/);
});
