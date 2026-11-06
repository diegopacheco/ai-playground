import { describe, expect, test } from "bun:test";
import { parseClientMessage, shellArgs } from "../src/terminal";

describe("parseClientMessage", () => {
  test("keystrokes pass through untouched, including control characters", () => {
    expect(parseClientMessage(JSON.stringify({ type: "input", data: "\u0003ls\r" }))).toEqual({ type: "input", data: "\u0003ls\r" });
  });

  test("resize carries the browser terminal size to the PTY", () => {
    expect(parseClientMessage(JSON.stringify({ type: "resize", cols: 132, rows: 40 }))).toEqual({ type: "resize", cols: 132, rows: 40 });
  });

  test("absurd sizes and garbage are ignored instead of crashing the PTY", () => {
    expect(parseClientMessage(JSON.stringify({ type: "resize", cols: 0, rows: 99999 }))).toBeNull();
    expect(parseClientMessage("not json")).toBeNull();
    expect(parseClientMessage(JSON.stringify({ type: "exec", data: "rm -rf /" }))).toBeNull();
  });
});

describe("shellArgs", () => {
  test("uses exec with a TTY, not connect, because exiting connect ends the sandbox main process", () => {
    expect(shellArgs("box")).toEqual(["openshell", "sandbox", "exec", "box", "--tty", "--", "bash", "-l"]);
  });
});
