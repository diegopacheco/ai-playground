import { describe, expect, test } from "bun:test";
import { ApiError } from "../../http";
import { parseActors, parseAgentState } from "./types";

describe("parsers", () => {
  test("a null actor list from Go means no actors", () => {
    expect(parseActors(null)).toEqual([]);
  });
  test("an unknown state is rejected instead of showing a wrong badge", () => {
    const actor = { name: "a", template: "t", state: "FLYING", workerPod: "", node: "", version: 1, snapshots: 0, createdAt: "" };
    expect(() => parseActors([actor])).toThrow(ApiError);
  });
  test("agent state with null note slices parses to empty lists", () => {
    const s = parseAgentState({ bootId: "b", startedAt: "x", hostname: "h", requests: 1, memoryNotes: null, journalNotes: null });
    expect(s.memoryNotes).toEqual([]);
  });
});
