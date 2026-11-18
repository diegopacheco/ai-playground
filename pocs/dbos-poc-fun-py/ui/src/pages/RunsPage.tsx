import { useState, type FormEvent } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { fetchRuns, startRun } from "../api.ts";
import { pillClass } from "../dbos.ts";
import { JournalButton } from "../JournalButton.tsx";

export function RunsPage() {
  const [goal, setGoal] = useState("Plan a 3 day trip to Lisbon for a food lover");
  const navigate = useNavigate();
  const runs = useQuery({ queryKey: ["runs"], queryFn: fetchRuns, refetchInterval: 3000 });
  const start = useMutation({ mutationFn: startRun, onSuccess: runId => navigate({ to: "/runs/$runId", params: { runId } }) });

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (goal.trim()) start.mutate(goal.trim());
  };

  return (
    <main>
      <form className="card goal" onSubmit={submit}>
        <label htmlFor="goal">Give the agent a goal</label>
        <textarea id="goal" value={goal} onChange={event => setGoal(event.target.value)} rows={3} />
        <button type="submit" disabled={start.isPending}>{start.isPending ? "Starting" : "Start agent workflow"}</button>
        {start.error && <p className="error">{start.error.message}</p>}
      </form>

      <section className="card">
        <h2>Workflow runs</h2>
        {runs.error && <p className="error">{runs.error.message}</p>}
        {runs.data?.length === 0 && <p className="muted">No runs yet.</p>}
        <table>
          <tbody>
            {runs.data?.map(run => (
              <tr key={run.run_id}>
                <td><Link to="/runs/$runId" params={{ runId: run.run_id }}>{run.run_id}</Link></td>
                <td><span className={`pill ${pillClass(run.status)}`}>{run.status}</span></td>
                <td className="muted">{new Date(run.created_at).toLocaleString()}</td>
                <td className="right"><JournalButton runId={run.run_id} label="Open DBOS journal" /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  );
}
