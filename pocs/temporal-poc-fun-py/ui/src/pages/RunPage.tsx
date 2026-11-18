import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { fetchRun } from "../api.ts";
import { isFinalRun, pillClass, progress } from "../temporal.ts";
import { TemporalButton } from "../TemporalButton.tsx";
import { runRoute } from "../router.tsx";

export function RunPage() {
  const { runId } = runRoute.useParams();
  const run = useQuery({
    queryKey: ["run", runId],
    queryFn: () => fetchRun(runId),
    refetchInterval: query => (isFinalRun(query.state.data) ? false : 1500),
  });
  const state = run.data?.state;

  return (
    <main>
      <Link to="/" className="back">All runs</Link>
      <section className="card">
        <div className="head">
          <div>
            <h2>{runId}</h2>
            {state && <p className="muted">{state.provider} / {state.model}</p>}
            {run.data && <p><span className={`pill ${pillClass(run.data.status)}`}>{run.data.status}</span></p>}
          </div>
          <TemporalButton url={run.data?.temporal_url} />
        </div>
        {run.error && <p className="error">{run.error.message}</p>}
        {!state && !run.error && <p className="muted">Waiting for the workflow to start.</p>}
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
