import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchClients, fetchLogsSummary, fetchRuns, fetchStateSummary } from "../api";
import { isAdmin, useAccess } from "../access";
import { nsToIso, percent, shortTime } from "../format";
import {
  CLI,
  COLOR,
  CONSUMER_X,
  HEIGHT,
  SERVICES,
  WIDTH,
  consumerEdge,
  consumerY,
  consumers,
  diff,
  pointAt,
  replayActivities,
  route,
  segmentPath,
  serviceEdge,
  servicePoint,
  snapshot,
  storeEdge,
  storePoint,
  usage,
  type Activity,
  type Capability,
  type Segment,
  type ServiceName,
  type Snapshot,
  type Usage,
} from "../topology";
import type { ClientInfo, LogLine } from "../types";

const POLL_MS = 2000;
const TRAVEL_MS = 1700;
const GLOW_MS = 2600;

type Particle = { id: number; segments: Segment[]; color: string; start: number; replay: boolean };

const REPLAY_EVERY_MS = 450;

const SERVICE_TEXT: Record<ServiceName, string> = {
  auth: "OAuth 2 tokens",
  state: "tenant k/v",
  logs: "per tenant logs",
  evals: "9 evaluators",
};

function useFlow(events: Activity[], replay: Activity[], yOf: (tenant: string) => number | null) {
  const [particles, setParticles] = useState<Particle[]>([]);
  const [glow, setGlow] = useState<Record<string, number>>({});
  const [now, setNow] = useState(() => performance.now());
  const nextId = useRef(0);

  useEffect(() => {
    if (events.length === 0) return;
    const born: Particle[] = [];
    const lit: Record<string, number> = {};
    const t = performance.now();
    for (const event of events) {
      const y = yOf(event.tenant);
      if (y === null) continue;
      const segments = route(y, event.capability);
      for (let i = 0; i < Math.min(event.count, 4); i++) {
        born.push({ id: nextId.current++, segments, color: COLOR[event.capability], start: t + i * 220, replay: false });
      }
      for (const key of [`c:${event.tenant}`, `s:${event.capability}`, `d:${event.capability}`]) lit[key] = t + GLOW_MS;
    }
    setParticles((current) => [...current, ...born]);
    setGlow((current) => ({ ...current, ...lit }));
  }, [events, yOf]);

  const replayRef = useRef(replay);
  replayRef.current = replay;
  useEffect(() => {
    let cursor = 0;
    const timer = setInterval(() => {
      const list = replayRef.current;
      if (list.length === 0) return;
      const event = list[cursor % list.length];
      cursor++;
      const y = yOf(event.tenant);
      if (y === null) return;
      const p = { id: nextId.current++, segments: route(y, event.capability), color: COLOR[event.capability], start: performance.now(), replay: true };
      setParticles((current) => [...current, p]);
    }, REPLAY_EVERY_MS);
    return () => clearInterval(timer);
  }, [yOf]);

  useEffect(() => {
    let frame = 0;
    const tick = () => {
      const t = performance.now();
      setNow(t);
      setParticles((current) => (current.some((p) => t - p.start > TRAVEL_MS) ? current.filter((p) => t - p.start <= TRAVEL_MS) : current));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  return { particles, glow, now };
}

function ConsumerNode({ client, y, use, lit }: { client: ClientInfo; y: number; use: Usage | undefined; lit: boolean }) {
  return (
    <g className={`node consumer kind-${client.kind} ${lit ? "lit" : ""}`} transform={`translate(${CONSUMER_X - 95},${y - 44})`}>
      <rect width="190" height="88" rx="14" />
      <text x="14" y="24" className="node-title">
        {client.client_id}
      </text>
      <text x="14" y="42" className="node-sub">
        {client.kind === "skill" ? "Claude Code skill" : "MCP server"}
      </text>
      <g transform="translate(14,54)">
        <Badge x={0} label={`${use?.keys ?? 0} keys`} on={(use?.keys ?? 0) > 0} color={COLOR.state} />
        <Badge x={62} label={`${use?.logs ?? 0} logs`} on={(use?.logs ?? 0) > 0} color={COLOR.logs} />
        <Badge x={124} label={use?.lastScore == null ? "no runs" : percent(use.lastScore)} on={use?.lastScore != null} color={COLOR.evals} />
      </g>
    </g>
  );
}

function Badge({ x, label, on, color }: { x: number; label: string; on: boolean; color: string }) {
  return (
    <g transform={`translate(${x},0)`} className={on ? "badge on" : "badge"}>
      <rect width="56" height="20" rx="10" style={on ? { fill: color } : undefined} />
      <text x="28" y="14" textAnchor="middle">
        {label}
      </text>
    </g>
  );
}

function Box({ x, y, w, h, title, sub, className }: { x: number; y: number; w: number; h: number; title: string; sub: string; className: string }) {
  return (
    <g className={`node ${className}`} transform={`translate(${x - w / 2},${y - h / 2})`}>
      <rect width={w} height={h} rx="14" />
      <text x={w / 2} y={h / 2 - 3} textAnchor="middle" className="node-title">
        {title}
      </text>
      <text x={w / 2} y={h / 2 + 16} textAnchor="middle" className="node-sub">
        {sub}
      </text>
    </g>
  );
}

function Graph({ list, usages, events, replay }: { list: ClientInfo[]; usages: Usage[]; events: Activity[]; replay: Activity[] }) {
  const ys = useMemo(() => new Map(list.map((c, i) => [c.client_id, consumerY(i, list.length)])), [list]);
  const yOf = useMemo(() => (tenant: string) => ys.get(tenant) ?? null, [ys]);
  const { particles, glow, now } = useFlow(events, replay, yOf);
  const isLit = (key: string) => (glow[key] ?? 0) > now;
  const capabilities: Capability[] = ["state", "logs", "evals"];

  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="graph" role="img" aria-label="Live topology of consumers and infra">
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <text x={CONSUMER_X} y="22" textAnchor="middle" className="col-title">
        SKILLS AND MCP SERVERS
      </text>
      <text x={CLI.x} y="22" textAnchor="middle" className="col-title">
        CLI
      </text>
      <text x={servicePoint("auth").x} y="22" textAnchor="middle" className="col-title">
        CAPABILITY APIS
      </text>
      <text x={storePoint("postgres").x} y="22" textAnchor="middle" className="col-title">
        STORES (PODMAN)
      </text>

      {list.map((c) => (
        <path key={c.client_id} d={segmentPath(consumerEdge(ys.get(c.client_id) ?? 0))} className={`edge ${isLit(`c:${c.client_id}`) ? "flow" : ""}`} />
      ))}
      {SERVICES.map((s) => (
        <path key={s} d={segmentPath(serviceEdge(s))} className={`edge ${s !== "auth" && isLit(`s:${s}`) ? "flow" : ""}`} />
      ))}
      {capabilities.map((c) => (
        <path key={c} d={segmentPath(storeEdge(c))} className={`edge ${isLit(`d:${c}`) ? "flow" : ""}`} />
      ))}

      {list.map((c) => (
        <ConsumerNode key={c.client_id} client={c} y={ys.get(c.client_id) ?? 0} use={usages.find((u) => u.tenant === c.client_id)} lit={isLit(`c:${c.client_id}`)} />
      ))}
      <Box x={CLI.x} y={CLI.y} w={170} h={90} title="agentinfra" sub="token cache, JSON out" className="cli" />
      {SERVICES.map((s) => (
        <Box key={s} x={servicePoint(s).x} y={servicePoint(s).y} w={170} h={70} title={s} sub={SERVICE_TEXT[s]} className={`svc svc-${s} ${s !== "auth" && isLit(`s:${s}`) ? "lit" : ""}`} />
      ))}
      <Box x={storePoint("postgres").x} y={storePoint("postgres").y} w={170} h={80} title="Postgres" sub="Row Level Security" className={`store ${isLit("d:state") || isLit("d:evals") ? "lit" : ""}`} />
      <Box x={storePoint("loki").x} y={storePoint("loki").y} w={170} h={80} title="Loki" sub="X-Scope-OrgID" className={`store ${isLit("d:logs") ? "lit" : ""}`} />

      {particles.map((p) => {
        const t = (now - p.start) / TRAVEL_MS;
        if (t < 0) return null;
        const point = pointAt(p.segments, t);
        return p.replay ? (
          <circle key={p.id} cx={point.x} cy={point.y} r="4" fill={p.color} opacity="0.45" />
        ) : (
          <circle key={p.id} cx={point.x} cy={point.y} r="7" fill={p.color} filter="url(#glow)" />
        );
      })}
    </svg>
  );
}

function Matrix({ list, usages }: { list: ClientInfo[]; usages: Usage[] }) {
  return (
    <table className="matrix">
      <thead>
        <tr>
          <th>Consumer</th>
          <th>Auth scopes</th>
          <th>State keys</th>
          <th>Log lines (1h)</th>
          <th>Warn + error</th>
          <th>Eval runs</th>
          <th>Last score</th>
        </tr>
      </thead>
      <tbody>
        {list.map((c) => {
          const u = usages.find((x) => x.tenant === c.client_id);
          return (
            <tr key={c.client_id}>
              <td>
                <span className="tenant">{c.client_id}</span> <span className={`kind kind-${c.kind}`}>{c.kind}</span>
              </td>
              <td className="mono">{c.scopes.length}</td>
              <td className={u?.keys ? "hot" : "cold"}>{u?.keys ?? 0}</td>
              <td className={u?.logs ? "hot" : "cold"}>{u?.logs ?? 0}</td>
              <td className={u?.warns ? "warnish" : "cold"}>{u?.warns ?? 0}</td>
              <td className={u?.runs ? "hot" : "cold"}>{u?.runs ?? 0}</td>
              <td>{u?.lastScore == null ? "-" : percent(u.lastScore)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function Feed({ lines }: { lines: LogLine[] }) {
  return (
    <ul className="feed">
      {lines.slice(0, 14).map((l) => (
        <li key={`${l.tenant}-${l.timestamp_ns}`} className={`feed-item feed-${l.level}`}>
          <span className="feed-head">
            <span className="feed-time mono">{shortTime(nsToIso(l.timestamp_ns))}</span>
            <span className="tenant">{l.tenant}</span>
          </span>
          <span className="feed-msg">{l.message}</span>
        </li>
      ))}
      {lines.length === 0 && <li className="muted">No log lines in the last hour. Run ./scripts/traffic.sh</li>}
    </ul>
  );
}

export function Overview() {
  const { session } = useAccess();
  if (!isAdmin(session.scopes)) {
    return (
      <section className="panel">
        <h2>Overview needs infra-admin</h2>
        <p className="muted">
          The live topology reads every tenant, so it needs <code>clients:read</code>, <code>state:read:all</code>,{" "}
          <code>logs:read:all</code> and <code>evals:read:all</code>. <span className="tenant">{session.clientId}</span> can open the
          Logs, State, Evals and Auth tabs and will see only its own data there.
        </p>
      </section>
    );
  }
  return <AdminOverview />;
}

function AdminOverview() {
  const clients = useQuery({ queryKey: ["clients"], queryFn: fetchClients });
  const state = useQuery({ queryKey: ["state-summary"], queryFn: fetchStateSummary, refetchInterval: POLL_MS });
  const logs = useQuery({ queryKey: ["logs-summary"], queryFn: fetchLogsSummary, refetchInterval: POLL_MS });
  const runs = useQuery({ queryKey: ["runs", "overview"], queryFn: () => fetchRuns(), refetchInterval: POLL_MS });
  const previous = useRef<Snapshot | null>(null);
  const [events, setEvents] = useState<Activity[]>([]);

  useEffect(() => {
    if (!state.data || !logs.data || !runs.data) return;
    const next = snapshot(state.data, logs.data, runs.data);
    const found = diff(previous.current, next);
    previous.current = next;
    if (found.length > 0) setEvents(found);
  }, [state.data, logs.data, runs.data]);

  const error = clients.error ?? state.error ?? logs.error ?? runs.error;
  if (error) return <div className="alert">{error.message}. The console needs infra-admin (clients:read, state:read:all, logs:read:all, evals:read:all).</div>;
  if (!clients.data || !state.data || !logs.data || !runs.data) return <p className="muted">Loading the infra</p>;

  const list = consumers(clients.data);
  const usages = usage(clients.data, state.data, logs.data, runs.data);
  const replay = replayActivities(state.data, logs.data, runs.data, Date.now());
  const totalKeys = state.data.reduce((s, x) => s + x.keys, 0);
  const totalLogs = logs.data.counts.reduce((s, x) => s + x.count, 0);

  return (
    <>
      <section className="tiles">
        <div className="tile">
          <span>Consumers</span>
          <b>{list.length}</b>
          <small className="muted">{list.filter((c) => c.kind === "skill").length} skills, {list.filter((c) => c.kind === "mcp").length} MCP servers</small>
        </div>
        <div className="tile">
          <span>State keys</span>
          <b style={{ color: COLOR.state }}>{totalKeys}</b>
          <small className="muted">{state.data.length} tenants in Postgres</small>
        </div>
        <div className="tile">
          <span>Log lines, last hour</span>
          <b style={{ color: COLOR.logs }}>{totalLogs}</b>
          <small className="muted">{new Set(logs.data.counts.map((c) => c.tenant)).size} Loki tenants</small>
        </div>
        <div className="tile">
          <span>Eval runs</span>
          <b style={{ color: COLOR.evals }}>{runs.data.length}</b>
          <small className="muted">graded by the evals API</small>
        </div>
      </section>
      <div className="overview">
        <section className="panel graph-panel">
          <div className="panel-head">
            <h2>Live topology</h2>
            <div className="legend">
              <span><i style={{ background: COLOR.state }} /> state write</span>
              <span><i style={{ background: COLOR.logs }} /> log line</span>
              <span><i style={{ background: COLOR.evals }} /> eval run</span>
              <span className="muted">bright = live, faint = replay of the last hour</span>
            </div>
          </div>
          <Graph list={list} usages={usages} events={events} replay={replay} />
        </section>
      </div>
      <div className="lower">
        <section className="panel">
          <h2>Which infra each consumer uses</h2>
          <Matrix list={list} usages={usages} />
        </section>
        <section className="panel feed-panel">
          <h2>Activity</h2>
          <Feed lines={logs.data.recent} />
        </section>
      </div>
    </>
  );
}
