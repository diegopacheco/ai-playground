import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import type { PolicyEvent } from "../types";
import { columnHelper, DataTable } from "./DataTable";

const col = columnHelper<PolicyEvent>();

const columns = col.columns([
  col.accessor("timeMs", { header: "Time", cell: (c) => new Date(c.getValue()).toLocaleTimeString() }),
  col.accessor("decision", { header: "Decision", cell: (c) => <span className={`badge ${c.getValue().toLowerCase()}`}>{c.getValue()}</span> }),
  col.accessor("kind", { header: "Event" }),
  col.accessor("target", { header: "Target", cell: (c) => <code>{c.getValue()}</code> }),
  col.accessor("policy", { header: "Rule" }),
  col.accessor("engine", { header: "Engine" }),
  col.accessor("reason", { header: "Reason", cell: (c) => <span className="reason">{c.getValue()}</span> }),
]);

export function DecisionLog({ name }: { name: string }) {
  const events = useQuery({ queryKey: ["events", name], queryFn: () => api.events(name), refetchInterval: 4000 });
  return (
    <section className="card">
      <h2>Decision log</h2>
      <p className="hint">OCSF audit events from the sandbox proxy. Every allowed or denied connection and HTTP request, with the rule and reason.</p>
      {events.error && <em className="error">{events.error.message}</em>}
      <DataTable data={events.data ?? []} columns={columns} empty="No decisions yet. Run a command in the console." rowClass={(e) => e.decision.toLowerCase()} />
    </section>
  );
}
