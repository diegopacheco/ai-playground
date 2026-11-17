import { describe, expect, test } from "bun:test";
import { route, triggers } from "./router";

describe("poteto-mode routes one sentence to one playbook", () => {
  test("a defect goes to Bug fix, not Feature, even when it says add", () => {
    expect(route("The contact form is broken, add a guard: it sends emails twice on double click")).toMatchObject({ kind: "playbook", id: "bug-fix" });
  });

  test("a read-only question goes to Investigation so no code is written", () => {
    expect(route("Explain how does the cache invalidation work")).toMatchObject({ kind: "playbook", id: "investigation" });
  });

  test("slowness goes to Perf issue", () => {
    expect(route("Search page got slow, latency doubled")).toMatchObject({ kind: "playbook", id: "perf-issue" });
  });

  test("large cross-cutting work escapes the playbooks to figure-it-out", () => {
    expect(route("Add auth and migrate every service across the codebase")).toEqual({ kind: "figure-it-out", matched: ["migrate", "across the codebase"] });
  });

  test("a task no playbook fits goes to figure-it-out instead of a wrong guess", () => {
    expect(route("hello there")).toEqual({ kind: "figure-it-out", matched: [] });
  });

  test("keywords match whole words only", () => {
    expect(route("format the address field")).toEqual({ kind: "figure-it-out", matched: [] });
  });
});

describe("fixed triggers fire from the task", () => {
  test("prose always runs unslop", () => {
    expect(triggers("anything").map((t) => t.skill)).toEqual(["unslop"]);
  });

  test("a perf task with a module change runs how, architect and benchmark-checklist", () => {
    expect(triggers("make the search module faster").map((t) => t.skill)).toEqual(["architect", "benchmark-checklist", "unslop"]);
    expect(triggers("fix slow search module").map((t) => t.skill)).toEqual(["how", "architect", "benchmark-checklist", "unslop"]);
  });
});
