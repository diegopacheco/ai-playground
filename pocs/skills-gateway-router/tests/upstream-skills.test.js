import { test } from "node:test";
import assert from "node:assert/strict";
import { countText } from "../upstream-skills/text-skills/word-count/word-count.js";
import { slugify } from "../upstream-skills/text-skills/slugify/slugify.js";
import { formatJson } from "../upstream-skills/dev-skills/json-format/json-format.js";

test("word-count does not count a trailing newline as an extra line", () => {
  assert.deepEqual(countText("one two\nthree\n"), { words: 3, lines: 2, characters: 14 });
  assert.deepEqual(countText(""), { words: 0, lines: 0, characters: 0 });
});

test("slugify strips accents and punctuation so the slug is URL safe", () => {
  assert.equal(slugify("  Olá, Mundo! Skills & Gateways  "), "ola-mundo-skills-gateways");
});

test("json-format indents valid JSON and throws on invalid JSON so the file is never rewritten", () => {
  assert.equal(formatJson('{"a":[1,2]}'), '{\n  "a": [\n    1,\n    2\n  ]\n}');
  assert.throws(() => formatJson("{a:1}"));
});
