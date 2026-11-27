import { expect, test } from "bun:test";
import { age } from "./format";

test("age rounds down to the largest unit and hides missing timestamps", () => {
  const now = Date.parse("2026-01-01T02:00:00Z");
  expect(age("2026-01-01T01:59:30Z", now)).toBe("30s");
  expect(age("2026-01-01T01:30:00Z", now)).toBe("30m");
  expect(age("2026-01-01T00:00:00Z", now)).toBe("2h");
  expect(age("0001-01-01T00:00:00Z", now)).toBe("-");
});
