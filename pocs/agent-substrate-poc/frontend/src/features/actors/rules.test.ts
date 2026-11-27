import { describe, expect, test } from "bun:test";
import { ACTOR_STATES } from "./types";
import { allowedActions, isTransient, memoryVerdict, nameError } from "./rules";

const note = (id: number) => ({ id, text: `n${id}`, at: "2026-01-01T00:00:00Z" });
const base = { bootId: "b", startedAt: "2026-01-01T00:00:00Z", hostname: "h", requests: 1 };

describe("allowedActions", () => {
  test("a running actor can be suspended but not resumed", () => {
    expect(allowedActions("RUNNING")).toEqual(["suspend", "delete"]);
  });
  test("a hibernated actor can be resumed", () => {
    expect(allowedActions("SUSPENDED")).toContain("resume");
  });
  test("no buttons while the actor is moving, so clicks cannot race the control plane", () => {
    for (const s of ACTOR_STATES.filter(isTransient)) expect(allowedActions(s)).toEqual([]);
  });
});

describe("memoryVerdict", () => {
  test("equal RAM and journal means the memory snapshot restored the process", () => {
    expect(memoryVerdict({ ...base, memoryNotes: [note(1), note(2)], journalNotes: [note(1), note(2)] })).toBe("restored");
  });
  test("RAM behind the journal means a cold boot that only kept the volume", () => {
    expect(memoryVerdict({ ...base, memoryNotes: [note(3)], journalNotes: [note(1), note(2), note(3)] })).toBe("cold-boot");
  });
  test("no notes yet", () => {
    expect(memoryVerdict({ ...base, memoryNotes: [], journalNotes: [] })).toBe("empty");
  });
});

describe("nameError", () => {
  test("accepts DNS-1123 labels and rejects what the control plane would reject", () => {
    expect(nameError("alice-2")).toBeUndefined();
    expect(nameError("")).toBeDefined();
    expect(nameError("Alice")).toBeDefined();
    expect(nameError("bad_name")).toBeDefined();
    expect(nameError("-x")).toBeDefined();
  });
});
