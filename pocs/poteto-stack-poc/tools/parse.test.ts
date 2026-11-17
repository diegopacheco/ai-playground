import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parsePlaybook, parsePrinciples } from "./parse";

const poteto = join(import.meta.dir, "..", ".claude", "skills", "poteto-mode");

describe("extractor reads the installed pstack files, so the UI never drifts from the skills", () => {
  test("playbook steps are copied word for word, in order, without markdown", () => {
    const pb = parsePlaybook("bug-fix", readFileSync(join(poteto, "playbooks", "bug-fix.md"), "utf8"));
    expect(pb.title).toBe("Bug fix");
    expect(pb.steps).toHaveLength(6);
    expect(pb.steps[0]).toStartWith("Reproduce it yourself");
    expect(pb.steps.join(" ")).not.toContain("**");
  });

  test("a playbook written as bullets still yields steps", () => {
    const pb = parsePlaybook("babysit", readFileSync(join(poteto, "playbooks", "babysit.md"), "utf8"));
    expect(pb.steps.length).toBeGreaterThan(0);
    expect(pb.steps[0]).toStartWith("Declare the mode");
  });

  test("every principle in poteto-mode is found with its group", () => {
    const principles = parsePrinciples(readFileSync(join(poteto, "SKILL.md"), "utf8"));
    expect(principles).toHaveLength(24);
    expect(principles.find((p) => p.name === "Prove It Works")?.group).toBe("Verification");
    expect(principles.find((p) => p.name === "Sequence Work into Verifiable Units")?.group).toBe("Verification");
    expect(principles.find((p) => p.name === "Encode Lessons in Structure")?.group).toBe("Meta");
  });
});
