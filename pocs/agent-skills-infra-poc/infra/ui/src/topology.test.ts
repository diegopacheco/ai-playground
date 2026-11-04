import { describe, expect, test } from "bun:test";
import { sortCredentials } from "./format";
import { consumers, diff, pointAt, replayActivities, route, snapshot, usage } from "./topology";
import type { ClientInfo, EvalRun, LogsSummary, StateSummary } from "./types";

const client = (client_id: string, kind: ClientInfo["kind"]): ClientInfo => ({ client_id, kind, description: "", scopes: [] });
const run = (id: number, tenant: string, score: number): EvalRun => ({ id, tenant, suite: "s", passed: 1, failed: 0, score, created_at: "2026-11-03T10:00:00Z" });
const logs = (counts: [string, string, number][]): LogsSummary => ({
  since_minutes: 60,
  counts: counts.map(([tenant, level, count]) => ({ tenant, level, count })),
  recent: [],
});
const state = (tenant: string, keys: number, last_updated: string): StateSummary => ({ tenant, keys, last_updated, recent_keys: [] });

describe("activity detection drives the animation, so it must only fire on real changes", () => {
  const before = snapshot([state("a", 1, "t1")], logs([["a", "info", 3]]), [run(1, "a", 1)]);

  test("the first poll animates nothing because there is no baseline", () => {
    expect(diff(null, before)).toEqual([]);
  });

  test("an unchanged poll animates nothing", () => {
    expect(diff(before, before)).toEqual([]);
  });

  test("a state write, new log lines and a new run each become one event for their tenant", () => {
    const after = snapshot([state("a", 2, "t2")], logs([["a", "info", 4], ["a", "warn", 1]]), [run(1, "a", 1), run(2, "a", 0.5)]);
    expect(diff(before, after)).toEqual([
      { tenant: "a", capability: "state", count: 1 },
      { tenant: "a", capability: "logs", count: 2 },
      { tenant: "a", capability: "evals", count: 1 },
    ]);
  });

  test("log lines ageing out of the window are not shown as activity", () => {
    const after = snapshot([state("a", 1, "t1")], logs([["a", "info", 1]]), [run(1, "a", 1)]);
    expect(diff(before, after)).toEqual([]);
  });

  test("activity of one tenant is never attributed to another", () => {
    const after = snapshot([state("a", 1, "t1"), state("b", 1, "t9")], logs([["a", "info", 3]]), [run(1, "a", 1)]);
    expect(diff(before, after)).toEqual([{ tenant: "b", capability: "state", count: 1 }]);
  });
});

describe("routing particles", () => {
  test("a log particle starts at the consumer and ends at Loki, not Postgres", () => {
    const path = route(140, "logs");
    expect(pointAt(path, 0)).toEqual({ x: 215, y: 140 });
    expect(pointAt(path, 1)).toEqual({ x: 905, y: 395 });
  });

  test("state and evals particles both end at Postgres", () => {
    expect(pointAt(route(140, "state"), 1)).toEqual(pointAt(route(300, "evals"), 1));
  });
});

test("usage counts only the tenant's own keys, logs, warnings and newest score", () => {
  const rows = usage(
    [client("a", "skill"), client("b", "mcp")],
    [state("a", 3, "t")],
    logs([["a", "info", 5], ["a", "warn", 2], ["b", "error", 1]]),
    [run(1, "a", 1), run(4, "a", 0.25), run(2, "b", 1)],
  );
  expect(rows).toEqual([
    { tenant: "a", keys: 3, logs: 7, warns: 2, runs: 2, lastScore: 0.25 },
    { tenant: "b", keys: 0, logs: 1, warns: 1, runs: 1, lastScore: 1 },
  ]);
});

test("the topology shows skills then MCP servers and hides admin and test clients", () => {
  const list = consumers([client("z-mcp", "mcp"), client("itest-alpha", "test"), client("infra-admin", "admin"), client("b-skill", "skill")]);
  expect(list.map((c) => c.client_id)).toEqual(["b-skill", "z-mcp"]);
});

test("the login panel lists admin clients first so the console login is one click", () => {
  const sorted = sortCredentials([client("itest-alpha", "test"), client("decision-log", "skill"), client("infra-admin", "admin")]);
  expect(sorted.map((c) => c.client_id)).toEqual(["infra-admin", "decision-log", "itest-alpha"]);
});

describe("replay keeps the graph moving with real recent activity only", () => {
  const now = Date.parse("2026-11-04T06:00:00Z");
  const ns = (iso: string) => String(BigInt(Date.parse(iso)) * 1_000_000n);
  const line = (tenant: string, iso: string) => ({ tenant, timestamp_ns: ns(iso), level: "info", message: "m", fields: {} });
  const summary = (lines: ReturnType<typeof line>[]): LogsSummary => ({ since_minutes: 60, counts: [], recent: lines });

  test("activity older than the window is never replayed, so an idle system stays still", () => {
    const old = replayActivities(
      [state("a", 1, "2026-11-04T04:00:00Z")],
      summary([line("a", "2026-11-04T04:30:00Z")]),
      [{ ...run(1, "a", 1), created_at: "2026-11-04T04:00:00Z" }],
      now,
    );
    expect(old).toEqual([]);
  });

  test("recent logs, state writes and eval runs are all replayed on their own capability", () => {
    const items = replayActivities(
      [state("a", 1, "2026-11-04T05:59:00Z")],
      summary([line("a", "2026-11-04T05:58:00Z")]),
      [{ ...run(1, "a", 1), created_at: "2026-11-04T05:57:00Z" }],
      now,
    );
    expect(items.map((i) => i.capability)).toEqual(["logs", "state", "evals"]);
  });

  test("apps are interleaved so a noisy app cannot hide a quiet one", () => {
    const lines = [line("noisy", "2026-11-04T05:59:00Z"), line("noisy", "2026-11-04T05:58:00Z"), line("noisy", "2026-11-04T05:57:00Z"), line("quiet", "2026-11-04T05:56:00Z")];
    const order = replayActivities([], summary(lines), [], now).map((i) => i.tenant);
    expect(order.slice(0, 2)).toEqual(["noisy", "quiet"]);
  });
});
