import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync, execSync } from "node:child_process";
import { existsSync, readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { tempRoot } from "./fixtures.js";

const ROOT = resolve(import.meta.dirname, "..");

function script(name, args, dirs) {
  const result = spawnSync(join(ROOT, "scripts", name), args, {
    encoding: "utf8",
    env: { ...process.env, CLAUDE_SKILLS_DIR: dirs.claude, CODEX_SKILLS_DIR: dirs.codex },
  });
  return { code: result.status, out: result.stdout, err: result.stderr };
}

function dirs() {
  const root = tempRoot();
  return { claude: join(root, "claude"), codex: join(root, "codex") };
}

test("the installed skill runs the gateway of this repo from any directory, for both agents", () => {
  const target = dirs();
  assert.equal(script("install.sh", [], target).code, 0);
  for (const agent of ["claude", "codex"]) {
    const skill = readFileSync(join(target[agent], "skills-gateway/SKILL.md"), "utf8");
    assert.doesNotMatch(skill, /CLAUDE_SKILL_DIR/);
    const command = skill.match(/^node ("[^"]+") <command>$/m)[1];
    assert.match(execSync(`node ${command} list`, { cwd: "/", encoding: "utf8" }), /text__slugify/);
  }
});

test("install twice and uninstall twice both succeed, and uninstall removes only the gateway", () => {
  const target = dirs();
  assert.equal(script("install.sh", ["claude"], target).code, 0);
  assert.equal(script("install.sh", ["claude"], target).code, 0);
  assert.equal(script("uninstall.sh", ["claude"], target).code, 0);
  assert.equal(script("uninstall.sh", ["claude"], target).code, 0);
  assert.ok(existsSync(target.claude));
  assert.ok(!existsSync(join(target.claude, "skills-gateway")));
});

test("a skills-gateway folder that is not ours is never overwritten or deleted", () => {
  const target = dirs();
  const foreign = join(target.claude, "skills-gateway");
  mkdirSync(foreign, { recursive: true });
  writeFileSync(join(foreign, "SKILL.md"), "---\nname: skills-gateway\ndescription: someone else\n---\n");
  assert.notEqual(script("install.sh", ["claude"], target).code, 0);
  assert.notEqual(script("uninstall.sh", ["claude"], target).code, 0);
  assert.match(readFileSync(join(foreign, "SKILL.md"), "utf8"), /someone else/);
});

test("an unknown agent fails loud", () => {
  const result = script("install.sh", ["cursor"], dirs());
  assert.notEqual(result.code, 0);
  assert.match(result.err, /unknown agent: cursor/);
});
