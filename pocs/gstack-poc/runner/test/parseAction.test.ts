import { describe, test, expect } from "bun:test";
import { parseAction } from "../src/parseAction.ts";

describe("parseAction", () => {
  test("parses click with role+name selector", () => {
    const result = parseAction(
      JSON.stringify({
        action: "click",
        selector: { kind: "role", role: "button", name: "Login" },
        reason: "click the login button to start the flow",
      }),
    );
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.action).toEqual({
        tool: "click",
        selector: { kind: "role", role: "button", name: "Login" },
        reason: "click the login button to start the flow",
      });
    }
  });

  test("parses type with placeholder selector", () => {
    const result = parseAction(
      JSON.stringify({
        action: "type",
        selector: { kind: "placeholder", text: "Username" },
        text: "standard_user",
        reason: "fill in the standard_user username from the prompt",
      }),
    );
    expect(result.ok).toBe(true);
    if (result.ok && result.action.tool === "type") {
      expect(result.action.text).toBe("standard_user");
      expect(result.action.selector.kind).toBe("placeholder");
    }
  });

  test("parses assert_text", () => {
    const result = parseAction(
      JSON.stringify({
        action: "assert_text",
        selector: { kind: "text", text: "Products" },
        text: "Products",
        reason: "verify the inventory page header",
      }),
    );
    expect(result.ok).toBe(true);
  });

  test("parses done", () => {
    const result = parseAction(
      JSON.stringify({ action: "done", reason: "test passed" }),
    );
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.action.tool).toBe("done");
  });

  test("strips code fences", () => {
    const result = parseAction(
      "```json\n" +
        JSON.stringify({ action: "done", reason: "fenced output" }) +
        "\n```",
    );
    expect(result.ok).toBe(true);
  });

  test("rejects missing reason", () => {
    const result = parseAction(
      JSON.stringify({ action: "done" }),
    );
    expect(result.ok).toBe(false);
    if (result.ok === false) expect(result.error).toMatch(/reason/);
  });

  test("rejects unknown action", () => {
    const result = parseAction(
      JSON.stringify({ action: "navigate", reason: "go somewhere" }),
    );
    expect(result.ok).toBe(false);
  });

  test("rejects malformed JSON", () => {
    const result = parseAction("this is not json");
    expect(result.ok).toBe(false);
  });

  test("rejects click without selector", () => {
    const result = parseAction(
      JSON.stringify({ action: "click", reason: "no target" }),
    );
    expect(result.ok).toBe(false);
  });

  test("rejects type without text", () => {
    const result = parseAction(
      JSON.stringify({
        action: "type",
        selector: { kind: "placeholder", text: "Username" },
        reason: "missing the text to type",
      }),
    );
    expect(result.ok).toBe(false);
  });

  test("rejects unknown selector kind", () => {
    const result = parseAction(
      JSON.stringify({
        action: "click",
        selector: { kind: "css", value: ".foo" },
        reason: "css not allowed",
      }),
    );
    expect(result.ok).toBe(false);
  });
});
