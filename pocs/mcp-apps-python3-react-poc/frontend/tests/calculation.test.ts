import { describe, expect, test } from "bun:test";
import { prettyStep, toOutcome } from "../src/shared/calculation";

describe("tool results become view outcomes", () => {
  test("structured content of a successful call is the calculation", () => {
    const calculation = { expression: "2+3", result: "5", steps: ["2 + 3 = 5"] };
    expect(toOutcome({ structuredContent: calculation, content: [] })).toEqual({ kind: "ok", calculation });
  });

  test("a tool error surfaces the server message instead of a stale result", () => {
    const outcome = toOutcome({ isError: true, content: [{ type: "text", text: "division by zero" }] });
    expect(outcome).toEqual({ kind: "error", message: "division by zero" });
  });

  test("malformed structured content is treated as an error, never rendered", () => {
    const outcome = toOutcome({ structuredContent: { expression: "1", result: 1, steps: [] } });
    expect(outcome).toEqual({ kind: "error", message: "calculation failed" });
  });
});

describe("steps read like a calculator tape", () => {
  test("server operators are shown with display symbols", () => {
    expect(prettyStep("6 / 2 = 3")).toBe("6 ÷ 2 = 3");
    expect(prettyStep("3 * 4 = 12")).toBe("3 × 4 = 12");
    expect(prettyStep("5 - 8 = -3")).toBe("5 − 8 = -3");
  });

  test("negative numbers keep their sign", () => {
    expect(prettyStep("-2 + -3 = -5")).toBe("-2 + -3 = -5");
  });
});
