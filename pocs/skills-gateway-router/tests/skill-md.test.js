import { test } from "node:test";
import assert from "node:assert/strict";
import { parseSkillMd } from "../lib/skill-md.js";

test("frontmatter gives the name and description the router uses to list and match a skill", () => {
  const { meta, body } = parseSkillMd("---\nname: slugify\ndescription: \"Makes: slugs\"\n---\n\nRun it.\n");
  assert.equal(meta.name, "slugify");
  assert.equal(meta.description, "Makes: slugs");
  assert.equal(body, "Run it.");
});

test("a SKILL.md without frontmatter is rejected so a broken skill never routes silently", () => {
  assert.throws(() => parseSkillMd("# no frontmatter\n"), /frontmatter/);
});
