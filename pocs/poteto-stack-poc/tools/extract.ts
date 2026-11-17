import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parsePlaybook, parsePrinciples } from "./parse";
import type { Catalog } from "../src/domain/types";

const root = join(import.meta.dir, "..");
const skills = join(root, ".claude", "skills");
const poteto = join(skills, "poteto-mode");
const playbookDir = join(poteto, "playbooks");

const catalog: Catalog = {
  version: readFileSync(join(skills, "PSTACK-VERSION"), "utf8").trim(),
  skills: readdirSync(skills, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name).sort(),
  playbooks: readdirSync(playbookDir)
    .filter((f) => f.endsWith(".md"))
    .sort()
    .map((f) => parsePlaybook(f.replace(/\.md$/, ""), readFileSync(join(playbookDir, f), "utf8")))
    .filter((p) => p.steps.length > 0),
  principles: parsePrinciples(readFileSync(join(poteto, "SKILL.md"), "utf8")),
};

writeFileSync(join(root, "src", "data", "pstack.json"), JSON.stringify(catalog, null, 2) + "\n");
console.log(`extracted ${catalog.skills.length} skills, ${catalog.playbooks.length} playbooks, ${catalog.principles.length} principles`);
