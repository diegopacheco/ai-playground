import { describe, expect, test } from "bun:test";
import { complete, gate, reopen, report, skip, startRun, togglePrinciple } from "./run";
import type { Playbook, PlaybookId } from "./types";

const playbook: Playbook = { id: "bug-fix" as PlaybookId, title: "Bug fix", steps: ["Reproduce", "Fix", "Verify"] };

describe("done needs evidence: Prove It Works", () => {
  test("a fresh run cannot report done", () => {
    const result = gate(startRun(playbook));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.missing).toContain("no evidence: Reproduce");
  });

  test("a step marked done with blank evidence stays todo", () => {
    expect(complete(startRun(playbook), 0, "   ").steps[0]?.state).toBe("todo");
  });

  test("evidence on every step still fails until a principle is named", () => {
    let run = startRun(playbook);
    run = complete(complete(complete(run, 0, "repro clicked twice"), 1, "diff"), 2, "repro passes");
    expect(gate(run)).toEqual({ ok: false, missing: ["no principle named"] });
    expect(() => report(run)).toThrow("no principle named");
  });
});

describe("skipped steps stay visible with skip:", () => {
  test("a skip needs a reason and shows in the report instead of vanishing", () => {
    let run = startRun(playbook);
    expect(skip(run, 1, "").steps[1]?.state).toBe("todo");
    run = togglePrinciple(complete(skip(complete(run, 0, "repro"), 1, "no code change needed"), 2, "repro passes"), "Prove It Works");
    const text = report(run);
    expect(text).toContain("skip: Fix | reason: no code change needed");
    expect(text.split("\n")).toHaveLength(5);
  });

  test("reopening a step blocks done again", () => {
    let run = togglePrinciple(startRun(playbook), "Fix Root Causes");
    run = complete(complete(complete(run, 0, "a"), 1, "b"), 2, "c");
    expect(gate(run).ok).toBe(true);
    expect(gate(reopen(run, 2)).ok).toBe(false);
  });
});
