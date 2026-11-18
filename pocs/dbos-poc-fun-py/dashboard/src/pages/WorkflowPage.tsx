import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { fetchWorkflow } from "../api.ts";
import { formatDuration, isFinal, isInternal, pillClass, preview, stepDuration, stepValue, timeline, workflowEnd } from "../dbos.ts";
import { workflowRoute } from "../router.tsx";
import { StoredView } from "../StoredView.tsx";
import { Timeline } from "../Timeline.tsx";
import type { WorkflowDetail } from "../types.ts";

type Tab = "timeline" | "steps" | "events";

const TABS: { id: Tab; label: string }[] = [
  { id: "timeline", label: "Timeline" },
  { id: "steps", label: "Steps" },
  { id: "events", label: "Events" },
];

const time = (ms: number | null) => (ms === null ? "-" : new Date(ms).toLocaleTimeString());

function Summary({ detail, now }: { detail: WorkflowDetail; now: number }) {
  const workflow = detail.workflow;
  const facts: [string, string][] = [
    ["Started", new Date(workflow.created_at).toLocaleString()],
    ["Ended", workflow.completed_at === null ? "-" : new Date(workflow.completed_at).toLocaleString()],
    ["Duration", formatDuration(workflowEnd(workflow, now) - workflow.created_at)],
    ["Recoveries", String(Math.max(0, workflow.recovery_attempts - 1))],
    ["Executor", workflow.executor || "-"],
    ["App version", workflow.app_version.slice(0, 12) || "-"],
    ["Queue", workflow.queue || "-"],
    ["Serialization", detail.serialization],
  ];
  return (
    <dl className="facts">
      {facts.map(([label, value]) => (
        <div key={label}><dt>{label}</dt><dd>{value}</dd></div>
      ))}
    </dl>
  );
}

function Steps({ detail, showInternal }: { detail: WorkflowDetail; showInternal: boolean }) {
  const steps = detail.steps.filter(step => showInternal || !isInternal(step));
  return (
    <table className="grid steps">
      <thead><tr><th>#</th><th>Step</th><th>Started</th><th>Duration</th><th>Recorded value</th></tr></thead>
      <tbody>
        {steps.map(step => {
          const value = stepValue(step, detail.history);
          const text = preview(value);
          return (
            <tr key={step.id} className={isInternal(step) ? "internal" : ""}>
              <td>{step.id}</td>
              <td className="mono">{step.name}{step.error.value !== null && <span className="pill error">error</span>}</td>
              <td className="muted">{time(step.started_at)}</td>
              <td>{stepDuration(step)}</td>
              <td>
                {text.length > 90 ? (
                  <details><summary>{text.slice(0, 90).replaceAll("\n", " ")}...</summary><StoredView stored={value} /></details>
                ) : (
                  <StoredView stored={value} empty="-" />
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function Events({ detail }: { detail: WorkflowDetail }) {
  return (
    <>
      <h3>Current events</h3>
      {detail.events.length === 0 && <p className="muted">No events set.</p>}
      {detail.events.map(event => (
        <div key={event.key} className="event">
          <span className="mono">{event.key}</span>
          <StoredView stored={event.value} />
        </div>
      ))}
      <h3>Event history</h3>
      <table className="grid">
        <thead><tr><th>Set by step</th><th>Key</th><th>Value</th></tr></thead>
        <tbody>
          {detail.history.map(event => (
            <tr key={`${event.function_id}-${event.key}`}>
              <td>{event.function_id}</td>
              <td className="mono">{event.key}</td>
              <td><details><summary>{preview(event.value).slice(0, 90).replaceAll("\n", " ")}</summary><StoredView stored={event.value} /></details></td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

export function WorkflowPage() {
  const { id } = workflowRoute.useParams();
  const [tab, setTab] = useState<Tab>("timeline");
  const [showInternal, setShowInternal] = useState(false);
  const query = useQuery({
    queryKey: ["workflow", id],
    queryFn: () => fetchWorkflow(id),
    refetchInterval: current => (isFinal(current.state.data?.workflow.status) ? false : 1500),
  });
  const detail = query.data;
  const now = Date.now();

  return (
    <main>
      <Link to="/" className="back">Workflows</Link>
      {query.error && <p className="error">{query.error.message}</p>}
      {detail && (
        <>
          <div className="title-row">
            <div>
              <h1 className="mono">{detail.workflow.id}</h1>
              <p className="muted">{detail.workflow.name}</p>
            </div>
            <span className={`pill big ${pillClass(detail.workflow.status)}`}>{detail.workflow.status}</span>
          </div>
          <section className="panel"><Summary detail={detail} now={now} /></section>
          <div className="io">
            <section className="panel"><h3>Input</h3><StoredView stored={detail.input} /></section>
            <section className="panel">
              <h3>{detail.error.value !== null ? "Error" : "Result"}</h3>
              <StoredView stored={detail.error.value !== null ? detail.error : detail.output} empty={isFinal(detail.workflow.status) ? "none" : "still running"} />
            </section>
          </div>
          <section className="panel">
            <div className="tabs">
              {TABS.map(item => (
                <button key={item.id} className={tab === item.id ? "tab active" : "tab"} onClick={() => setTab(item.id)}>{item.label}</button>
              ))}
              <label className="toggle">
                <input type="checkbox" checked={showInternal} onChange={event => setShowInternal(event.target.checked)} />
                show DBOS internal operations
              </label>
            </div>
            {tab === "timeline" && (
              <Timeline bars={timeline(detail, now, showInternal)} total={formatDuration(workflowEnd(detail.workflow, now) - detail.workflow.created_at)} />
            )}
            {tab === "steps" && <Steps detail={detail} showInternal={showInternal} />}
            {tab === "events" && <Events detail={detail} />}
          </section>
        </>
      )}
    </main>
  );
}
