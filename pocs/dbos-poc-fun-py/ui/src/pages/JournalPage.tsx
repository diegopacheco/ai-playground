import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { fetchJournal } from "../api.ts";
import { isAgentStep, isFinalWorkflow, pillClass, stepDuration } from "../dbos.ts";
import { journalRoute } from "../router.tsx";

export function JournalPage() {
  const { runId } = journalRoute.useParams();
  const journal = useQuery({
    queryKey: ["journal", runId],
    queryFn: () => fetchJournal(runId),
    refetchInterval: query => (isFinalWorkflow(query.state.data?.status) ? false : 1500),
  });
  const data = journal.data;

  return (
    <main>
      <Link to="/runs/$runId" params={{ runId }} className="back">Back to the run</Link>
      <section className="card">
        <div className="head">
          <div>
            <h2>DBOS journal of {runId}</h2>
            <p className="muted">workflow_status and operation_outputs rows from the DBOS system database</p>
          </div>
          {data && <span className={`pill ${pillClass(data.status)}`}>{data.status}</span>}
        </div>
        {journal.error && <p className="error">{journal.error.message}</p>}
        {data && (
          <>
            <p className="muted">executor {data.executor} · app version {data.app_version} · {data.steps.length} recorded steps</p>
            <table className="journal">
              <thead>
                <tr><th>#</th><th>step</th><th>took</th><th>recorded output</th></tr>
              </thead>
              <tbody>
                {data.steps.map(step => (
                  <tr key={step.id} className={isAgentStep(step) ? "agent" : "internal"}>
                    <td>{step.id}</td>
                    <td><code>{step.name}</code></td>
                    <td className="muted">{stepDuration(step)}</td>
                    <td>
                      {step.error && <p className="error">{step.error}</p>}
                      {step.output && <details><summary>{step.output.slice(0, 90)}</summary><pre>{step.output}</pre></details>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </section>
    </main>
  );
}
