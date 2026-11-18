import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { fetchStatus } from "../api.ts";
import { isFinal, progress } from "../dbos.ts";
import { JournalButton } from "../JournalButton.tsx";
import { runRoute } from "../router.tsx";

export function RunPage() {
  const { runId } = runRoute.useParams();
  const status = useQuery({
    queryKey: ["status", runId],
    queryFn: () => fetchStatus(runId),
    refetchInterval: query => (isFinal(query.state.data) ? false : 1500),
  });
  const state = status.data;

  return (
    <main>
      <Link to="/" className="back">All runs</Link>
      <section className="card">
        <div className="head">
          <div>
            <h2>{runId}</h2>
            {state && <p className="muted">{state.provider} / {state.model}</p>}
          </div>
          <JournalButton runId={runId} />
        </div>
        {status.error && <p className="error">{status.error.message}</p>}
        {!state && !status.error && <p className="muted">Waiting for the workflow to start.</p>}
        {state && (
          <>
            <p className="goal-text">{state.goal}</p>
            <div className="bar"><div style={{ width: `${progress(state)}%` }} /></div>
            <p><span className={`pill ${state.phase}`}>{state.phase}</span></p>
            <ol className="steps">
              {state.steps.map((step, index) => (
                <li key={index} className={step.status}>
                  <strong>{step.task}</strong>
                  <span className={`pill ${step.status}`}>{step.status}</span>
                  {step.output && <p>{step.output}</p>}
                </li>
              ))}
            </ol>
            {state.summary && <div className="summary"><h3>Final answer</h3><p>{state.summary}</p></div>}
            {state.error && <p className="error">{state.error}</p>}
          </>
        )}
      </section>
    </main>
  );
}
