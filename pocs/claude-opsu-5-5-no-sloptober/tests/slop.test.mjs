import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { score, commentRatio } from "../web/slop.mjs";

const skill = fileURLToPath(new URL("../.claude/skills/no-slop-pr/", import.meta.url));
const rules = JSON.parse(readFileSync(skill + "rules.json", "utf8"));
const read = (name) => readFileSync(skill + "tests/fixtures/" + name, "utf8");

function python(textFile, diffFile) {
  const args = [skill + "slop_check.py", "--json", "--text", skill + "tests/fixtures/" + textFile];
  if (diffFile) args.push("--diff", skill + "tests/fixtures/" + diffFile);
  try {
    return JSON.parse(execFileSync("python3", args, { encoding: "utf8" }));
  } catch (err) {
    return JSON.parse(err.stdout);
  }
}

const cases = [
  ["slop.txt", "slop.diff"],
  ["suspect.txt", undefined],
  ["human.txt", "human.diff"],
];

for (const [text, diff] of cases) {
  test(`page meter gives the same verdict as the skill for ${text}`, () => {
    const page = score(rules, read(text), diff ? read(diff) : "");
    assert.deepEqual(page, python(text, diff));
  });
}

test("agent PR is blocked in the page so the meter never contradicts the gate", () => {
  assert.equal(score(rules, read("slop.txt"), read("slop.diff")).verdict, "BLOCK");
});

test("a human bug fix is not punished", () => {
  assert.equal(score(rules, read("human.txt"), read("human.diff")).score, 0);
});

test("emoji rules match real emoji after json decoding", () => {
  assert.ok(score(rules, "## ✨ Summary\n- ✅ done").hits.some((h) => h.id === "emoji-headings"));
});

test("comment cruft ignores tiny diffs", () => {
  assert.equal(commentRatio("+# a\n+# b"), 0);
});
