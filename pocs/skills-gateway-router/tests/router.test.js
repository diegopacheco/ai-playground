import { test } from "node:test";
import assert from "node:assert/strict";
import { join } from "node:path";
import { createRouter, parseSources, splitSkillName, EXIT } from "../router.js";
import { skillFolder, skillMd, tempRoot } from "./fixtures.js";

function twoSourcesWithSameSkillName() {
  const root = tempRoot();
  skillFolder(root, "a", "summarize", skillMd("summarize", "Summarizes a meeting transcript.", "A body"));
  skillFolder(root, "b", "summarize", skillMd("summarize", "Summarizes a pull request diff.", "B body"));
  return { root, router: createRouter([{ name: "alpha", dir: join(root, "a") }, { name: "beta", dir: join(root, "b") }]) };
}

test("namespacing keeps two sources with the same skill name from colliding", () => {
  const { router } = twoSourcesWithSameSkillName();
  assert.deepEqual(router.catalog().skills.map((skill) => skill.name), ["alpha__summarize", "beta__summarize"]);
  assert.match(router.load("alpha__summarize"), /A body/);
  assert.match(router.load("beta__summarize"), /B body/);
});

test("the qualified name is split at the first separator only", () => {
  assert.deepEqual(splitSkillName("text__word-count"), { source: "text", skill: "word-count" });
  assert.equal(splitSkillName("word-count"), null);
  assert.equal(splitSkillName("__x"), null);
});

test("source names that could break the separator or repeat are rejected at startup", () => {
  assert.throws(() => parseSources("a__b=dir", "/r"), /invalid skill source/);
  assert.throws(() => parseSources("text=x,text=y", "/r"), /duplicate skill source: text/);
  assert.throws(() => parseSources("", "/r"), /no skill sources/);
  assert.deepEqual(parseSources("text=skills/text\ndev=/abs/dev\n", "/r"), [
    { name: "text", dir: "/r/skills/text" },
    { name: "dev", dir: "/abs/dev" },
  ]);
});

test("routing ranks the skill whose description matches the request first", () => {
  const { router } = twoSourcesWithSameSkillName();
  const { matches } = router.route("summarize the pull request diff");
  assert.equal(matches[0].name, "beta__summarize");
  assert.ok(matches[0].score > matches[1].score);
});

test("an unrelated request routes to nothing instead of guessing a skill", () => {
  const { router } = twoSourcesWithSameSkillName();
  assert.deepEqual(router.route("bake a chocolate cake").matches, []);
  assert.throws(() => router.route("a b"), (error) => error.code === EXIT.INVALID);
});

test("a dead source drops out of the catalog without hiding the healthy ones", () => {
  const root = tempRoot();
  skillFolder(root, "a", "ok", skillMd("ok", "Still works.", "body"));
  const router = createRouter([{ name: "alpha", dir: join(root, "a") }, { name: "gone", dir: join(root, "missing") }]);
  const { skills, errors } = router.catalog();
  assert.deepEqual(skills.map((skill) => skill.name), ["alpha__ok"]);
  assert.match(errors[0], /skill source unreachable: gone/);
  assert.deepEqual(router.status().map((source) => source.status), ["UP", "DOWN"]);
});

test("loading from a dead source fails loud with the unreachable code", () => {
  const router = createRouter([{ name: "gone", dir: "/nowhere/at/all" }]);
  assert.throws(() => router.load("gone__ok"), (error) => error.code === EXIT.UNREACHABLE && /unreachable/.test(error.message));
});

test("unknown sources and skills fail with the invalid code", () => {
  const { router } = twoSourcesWithSameSkillName();
  assert.throws(() => router.load("nope__summarize"), (error) => error.code === EXIT.INVALID && /no skill source/.test(error.message));
  assert.throws(() => router.load("alpha__nope"), (error) => error.code === EXIT.INVALID && /unknown skill/.test(error.message));
});

test("a broken skill is reported but its healthy siblings still route", () => {
  const root = tempRoot();
  skillFolder(root, "a", "good", skillMd("good", "Good skill.", "body"));
  skillFolder(root, "a", "nodesc", "---\nname: nodesc\n---\nbody\n");
  const { skills, errors } = createRouter([{ name: "alpha", dir: join(root, "a") }]).catalog();
  assert.deepEqual(skills.map((skill) => skill.name), ["alpha__good"]);
  assert.match(errors[0], /no description/);
});

test("load points the skill at its own folder and passes the arguments, so its scripts run from anywhere", () => {
  const root = tempRoot();
  const dir = skillFolder(root, "a", "run", skillMd("run", "Runs a script.", "Request: $ARGUMENTS\nnode ${SKILL_DIR}/run.js"));
  const text = createRouter([{ name: "alpha", dir: join(root, "a") }]).load("alpha__run", "hello there");
  assert.match(text, new RegExp(`base directory: ${dir}`));
  assert.ok(text.includes(`node ${dir}/run.js`));
  assert.ok(text.includes("Request: hello there"));
});
