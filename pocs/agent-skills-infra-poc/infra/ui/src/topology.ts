import type { ClientInfo, EvalRun, LogsSummary, StateSummary } from "./types";

export type Point = { x: number; y: number };
export type Segment = { from: Point; ctrl: Point; to: Point };
export type Capability = "state" | "logs" | "evals";
export type ServiceName = "auth" | Capability;
export type StoreName = "postgres" | "loki";
export type Activity = { tenant: string; capability: Capability; count: number };

export type Snapshot = {
  stateUpdated: Record<string, string>;
  logCounts: Record<string, number>;
  lastRun: Record<string, number>;
};

export type Usage = {
  tenant: string;
  keys: number;
  logs: number;
  warns: number;
  runs: number;
  lastScore: number | null;
};

export const WIDTH = 1120;
export const HEIGHT = 560;
export const SERVICES: ServiceName[] = ["auth", "state", "logs", "evals"];
export const STORE_OF: Record<Capability, StoreName> = { state: "postgres", logs: "loki", evals: "postgres" };
export const COLOR: Record<Capability, string> = { state: "#2f9e44", logs: "#0c8599", evals: "#e8590c" };

const SERVICE_Y: Record<ServiceName, number> = { auth: 95, state: 215, logs: 335, evals: 455 };
const STORE_Y: Record<StoreName, number> = { postgres: 215, loki: 395 };
export const CLI: Point = { x: 400, y: 280 };
export const SERVICE_X = 690;
export const STORE_X = 990;
export const CONSUMER_X = 120;

export function consumerY(index: number, total: number): number {
  const gap = HEIGHT / (total + 1);
  return Math.round(gap * (index + 1));
}

export function servicePoint(name: ServiceName): Point {
  return { x: SERVICE_X, y: SERVICE_Y[name] };
}

export function storePoint(name: StoreName): Point {
  return { x: STORE_X, y: STORE_Y[name] };
}

export function curve(from: Point, to: Point, bend = 18): Segment {
  return { from, to, ctrl: { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 - bend } };
}

export function segmentPath(s: Segment): string {
  return `M${s.from.x},${s.from.y} Q${s.ctrl.x},${s.ctrl.y} ${s.to.x},${s.to.y}`;
}

export function consumerEdge(y: number): Segment {
  return curve({ x: CONSUMER_X + 95, y }, { x: CLI.x - 85, y: CLI.y }, 10);
}

export function serviceEdge(name: ServiceName): Segment {
  const p = servicePoint(name);
  return curve({ x: CLI.x + 85, y: CLI.y }, { x: p.x - 85, y: p.y }, 10);
}

export function storeEdge(name: Capability): Segment {
  const from = servicePoint(name);
  const to = storePoint(STORE_OF[name]);
  return curve({ x: from.x + 85, y: from.y }, { x: to.x - 85, y: to.y }, 12);
}

export function route(consumerYValue: number, capability: Capability): Segment[] {
  return [consumerEdge(consumerYValue), serviceEdge(capability), storeEdge(capability)];
}

function bezier(s: Segment, t: number): Point {
  const u = 1 - t;
  return {
    x: u * u * s.from.x + 2 * u * t * s.ctrl.x + t * t * s.to.x,
    y: u * u * s.from.y + 2 * u * t * s.ctrl.y + t * t * s.to.y,
  };
}

export function pointAt(segments: Segment[], t: number): Point {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * segments.length;
  const index = Math.min(segments.length - 1, Math.floor(scaled));
  return bezier(segments[index], scaled - index);
}

export function snapshot(state: StateSummary[], logs: LogsSummary, runs: EvalRun[]): Snapshot {
  const logCounts: Record<string, number> = {};
  for (const row of logs.counts) logCounts[row.tenant] = (logCounts[row.tenant] ?? 0) + row.count;
  const lastRun: Record<string, number> = {};
  for (const run of runs) lastRun[run.tenant] = Math.max(lastRun[run.tenant] ?? 0, run.id);
  return { stateUpdated: Object.fromEntries(state.map((s) => [s.tenant, s.last_updated])), logCounts, lastRun };
}

export function diff(previous: Snapshot | null, next: Snapshot): Activity[] {
  if (previous === null) return [];
  const events: Activity[] = [];
  for (const [tenant, updated] of Object.entries(next.stateUpdated)) {
    if (previous.stateUpdated[tenant] !== updated) events.push({ tenant, capability: "state", count: 1 });
  }
  for (const [tenant, count] of Object.entries(next.logCounts)) {
    const added = count - (previous.logCounts[tenant] ?? 0);
    if (added > 0) events.push({ tenant, capability: "logs", count: added });
  }
  for (const [tenant, id] of Object.entries(next.lastRun)) {
    if (id > (previous.lastRun[tenant] ?? 0)) events.push({ tenant, capability: "evals", count: 1 });
  }
  return events;
}

export function usage(clients: ClientInfo[], state: StateSummary[], logs: LogsSummary, runs: EvalRun[]): Usage[] {
  return clients.map((c) => {
    const own = runs.filter((r) => r.tenant === c.client_id).sort((a, b) => b.id - a.id);
    const lines = logs.counts.filter((l) => l.tenant === c.client_id);
    return {
      tenant: c.client_id,
      keys: state.find((s) => s.tenant === c.client_id)?.keys ?? 0,
      logs: lines.reduce((sum, l) => sum + l.count, 0),
      warns: lines.filter((l) => l.level === "warn" || l.level === "error").reduce((sum, l) => sum + l.count, 0),
      runs: own.length,
      lastScore: own.length > 0 ? own[0].score : null,
    };
  });
}

export function consumers(clients: ClientInfo[]): ClientInfo[] {
  const order = { skill: 0, mcp: 1, admin: 2, test: 3, unknown: 4 };
  return clients
    .filter((c) => c.kind === "skill" || c.kind === "mcp")
    .sort((a, b) => order[a.kind] - order[b.kind] || a.client_id.localeCompare(b.client_id));
}

export const REPLAY_WINDOW_MS = 60 * 60 * 1000;

export function replayActivities(
  state: StateSummary[],
  logs: LogsSummary,
  runs: EvalRun[],
  nowMs: number,
  windowMs = REPLAY_WINDOW_MS,
): Activity[] {
  const recent = (iso: string) => nowMs - Date.parse(iso) <= windowMs;
  const items: Activity[] = [];
  for (const line of logs.recent) {
    if (nowMs - Number(BigInt(line.timestamp_ns) / 1_000_000n) <= windowMs) items.push({ tenant: line.tenant, capability: "logs", count: 1 });
  }
  for (const s of state) if (recent(s.last_updated)) items.push({ tenant: s.tenant, capability: "state", count: 1 });
  for (const r of runs) if (recent(r.created_at)) items.push({ tenant: r.tenant, capability: "evals", count: 1 });
  const byTenant = new Map<string, Activity[]>();
  for (const item of items) byTenant.set(item.tenant, [...(byTenant.get(item.tenant) ?? []), item]);
  const lanes = [...byTenant.values()];
  const interleaved: Activity[] = [];
  for (let i = 0; lanes.some((lane) => i < lane.length); i++) {
    for (const lane of lanes) if (i < lane.length) interleaved.push(lane[i]);
  }
  return interleaved;
}
