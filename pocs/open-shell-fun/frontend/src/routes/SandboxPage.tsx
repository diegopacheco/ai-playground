import { useQuery } from "@tanstack/react-query";
import { Link, Outlet } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { api } from "../api";
import { AttackLab } from "../components/AttackLab";
import { DecisionLog } from "../components/DecisionLog";
import { ExecConsole } from "../components/ExecConsole";
import { LiveFeed } from "../components/LiveFeed";
import { PolicyPanel } from "../components/PolicyPanel";
import { WebTerminal } from "../components/WebTerminal";

const TABS = [
  { to: "/sandboxes/$name", label: "Terminal" },
  { to: "/sandboxes/$name/attacks", label: "Attack Lab" },
  { to: "/sandboxes/$name/policy", label: "Policy" },
  { to: "/sandboxes/$name/log", label: "Decision log" },
] as const;

export function SandboxLayout({ name }: { name: string }) {
  const sandboxes = useQuery({ queryKey: ["sandboxes"], queryFn: api.sandboxes, refetchInterval: 3000 });
  const phase = sandboxes.data?.find((s) => s.name === name)?.phase ?? "unknown";
  return (
    <>
      <div className="crumbs">
        <Link to="/">Sandboxes</Link> / <strong>{name}</strong>
        <span className={`badge phase-${phase.toLowerCase()}`}>{phase}</span>
      </div>
      <nav className="tabs">
        {TABS.map((tab) => (
          <Link key={tab.to} to={tab.to} params={{ name }} activeOptions={{ exact: true }} activeProps={{ className: "active" }}>
            {tab.label}
          </Link>
        ))}
      </nav>
      <Outlet />
    </>
  );
}

function WhenReady({ name, children }: { name: string; children: ReactNode }) {
  const sandboxes = useQuery({ queryKey: ["sandboxes"], queryFn: api.sandboxes, refetchInterval: 3000 });
  const phase = sandboxes.data?.find((s) => s.name === name)?.phase;
  if (phase !== "Ready") {
    return <section className="card"><p className="empty">Waiting for the sandbox to be Ready (now: {phase ?? "loading"})...</p></section>;
  }
  return <>{children}</>;
}

export function TerminalTab({ name }: { name: string }) {
  return (
    <WhenReady name={name}>
      <div className="terminal-layout">
        <WebTerminal name={name} />
        <LiveFeed name={name} />
      </div>
    </WhenReady>
  );
}

export function AttacksTab({ name }: { name: string }) {
  return <WhenReady name={name}><AttackLab name={name} /></WhenReady>;
}

export function PolicyTab({ name }: { name: string }) {
  return (
    <div className="split">
      <ExecConsole name={name} />
      <PolicyPanel name={name} />
    </div>
  );
}

export function LogTab({ name }: { name: string }) {
  return <DecisionLog name={name} />;
}
