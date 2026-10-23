import { describe, expect, test } from "bun:test";
import { KEYS, keyForInput, press, type Key } from "../src/view/keypad";

const key = (label: string): Key => {
  const found = KEYS.find((candidate) => candidate.label === label);
  if (!found) throw new Error(`no key ${label}`);
  return found;
};

const typeKeys = (...labels: string[]) => labels.reduce((expression, label) => press(expression, key(label)), "");

describe("keypad builds expressions the server can parse", () => {
  test("digits and operators append in order", () => {
    expect(typeKeys("1", "2", "+", "3")).toBe("12+3");
  });

  test("a second operator replaces the first so the expression never has dangling operators", () => {
    expect(typeKeys("4", "+", "×", "2")).toBe("4×2");
  });

  test("binary operators are ignored at the start but minus is allowed for negative numbers", () => {
    expect(typeKeys("×")).toBe("");
    expect(typeKeys("−", "5")).toBe("−5");
    expect(typeKeys("(", "−", "2", ")")).toBe("(−2)");
  });

  test("a number cannot get two decimal points", () => {
    expect(typeKeys("1", ".", "5", ".", "2")).toBe("1.52");
  });

  test("a leading decimal point becomes 0. so the server never sees a bare dot", () => {
    expect(typeKeys(".", "5")).toBe("0.5");
    expect(typeKeys("2", "+", ".", "5")).toBe("2+0.5");
  });

  test("a new number after an operator can take its own decimal point", () => {
    expect(typeKeys("1", ".", "5", "+", "2", ".", "5")).toBe("1.5+2.5");
  });

  test("backspace removes one character and clear empties", () => {
    expect(press("12+3", key("⌫"))).toBe("12+");
    expect(press("12+3", key("C"))).toBe("");
  });

  test("equals leaves evaluation to the server", () => {
    expect(press("1+1", key("="))).toBe("1+1");
  });
});

describe("physical keyboard maps to keypad keys", () => {
  test("ascii operators map to display symbols", () => {
    expect(keyForInput("*")?.label).toBe("×");
    expect(keyForInput("/")?.label).toBe("÷");
    expect(keyForInput("-")?.label).toBe("−");
  });

  test("enter evaluates, escape clears, backspace deletes", () => {
    expect(keyForInput("Enter")?.kind).toBe("equals");
    expect(keyForInput("Escape")?.kind).toBe("clear");
    expect(keyForInput("Backspace")?.kind).toBe("back");
  });

  test("letters are not calculator keys", () => {
    expect(keyForInput("a")).toBeUndefined();
  });
});
