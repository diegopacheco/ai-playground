import { useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { fetchWorkflows } from "../api.ts";
import { formatDuration, pillClass, workflowEnd } from "../dbos.ts";

export function WorkflowsPage() {
  const [status, setStatus] = useState("");
  const [search, setSearch] = useState("");
  const list = useQuery({
    queryKey: ["workflows", status, search],
    queryFn: () => fetchWorkflows({ status, search }),
    refetchInterval: 2000,
    placeholderData: keepPreviousData,
  });
  const counts = list.data?.counts ?? [];
  const total = counts.reduce((sum, item) => sum + item.count, 0);
  const now = Date.now();

  return (
    <main>
      <div className="title-row">
        <h1>Workflows</h1>
        <input className="search" placeholder="Search by workflow id or name" value={search} onChange={event => setSearch(event.target.value)} />
      </div>
      <div className="filters">
        <button className={status === "" ? "chip active" : "chip"} onClick={() => setStatus("")}>All <b>{total}</b></button>
        {counts.map(item => (
          <button key={item.status} className={status === item.status ? "chip active" : "chip"} onClick={() => setStatus(item.status)}>
            <span className={`dot ${pillClass(item.status)}`} />{item.status} <b>{item.count}</b>
          </button>
        ))}
      </div>
      {list.error && <p className="error">{list.error.message}</p>}
      <section className="panel">
        <table className="grid">
          <thead>
            <tr><th>Status</th><th>Workflow ID</th><th>Name</th><th>Started</th><th>Duration</th><th>Steps</th><th>Recoveries</th></tr>
          </thead>
          <tbody>
            {list.data?.workflows.map(workflow => (
              <tr key={workflow.id}>
                <td><span className={`pill ${pillClass(workflow.status)}`}>{workflow.status}</span></td>
                <td><Link to="/workflows/$id" params={{ id: workflow.id }} className="mono">{workflow.id}</Link></td>
                <td>{workflow.name}</td>
                <td className="muted">{new Date(workflow.created_at).toLocaleString()}</td>
                <td>{formatDuration(workflowEnd(workflow, now) - workflow.created_at)}</td>
                <td>{workflow.steps}</td>
                <td>{Math.max(0, workflow.recovery_attempts - 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {list.data?.workflows.length === 0 && <p className="muted empty">No workflows match.</p>}
      </section>
    </main>
  );
}
