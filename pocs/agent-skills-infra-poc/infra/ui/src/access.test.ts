import { expect, test } from "bun:test";
import { isAdmin, viewOf } from "./access";
import { decodeClaims, nsToIso } from "./format";

const APP = ["state:read", "state:write", "logs:read", "logs:write", "evals:read", "evals:write"];
const ADMIN = ["clients:read", "state:read:all", "logs:read:all", "evals:read:all"];

test("an app client gets its own view on every tab and no app filter", () => {
  expect(["state", "logs", "evals"].map((c) => viewOf(APP, c as "state"))).toEqual(["own", "own", "own"]);
  expect(isAdmin(APP)).toBe(false);
});

test("infra-admin gets the all-apps view and the filter", () => {
  expect(["state", "logs", "evals"].map((c) => viewOf(ADMIN, c as "state"))).toEqual(["all", "all", "all"]);
  expect(isAdmin(ADMIN)).toBe(true);
});

test("a client without a read scope gets no view, so the tab explains the 403 instead of calling the API", () => {
  expect(viewOf(["evals:read:all"], "logs")).toBe("none");
  expect(viewOf(["state:read"], "evals")).toBe("none");
});

test("read:all wins over read when a client holds both", () => {
  expect(viewOf(["evals:read", "evals:read:all"], "evals")).toBe("all");
});

test("token claims are decoded from base64url so the Auth tab shows the real sub", () => {
  const payload = btoa(JSON.stringify({ sub: "decision-log", scope: "state:read" })).replace(/=+$/, "").replace(/\+/g, "-").replace(/\//g, "_");
  expect(decodeClaims(`h.${payload}.s`)).toEqual({ sub: "decision-log", scope: "state:read" });
  expect(decodeClaims("garbage")).toEqual({});
});

test("loki nanosecond timestamps keep millisecond precision", () => {
  expect(nsToIso("1793772260220252000")).toBe("2026-11-04T06:04:20.220Z");
});
