import { useMutation } from "@tanstack/react-query";
import { applyCbConfig, tuneCb } from "../api";
import type { CircuitBreakerSettings, RunSummary, TuneStatus } from "../types";
import { SETTING_KEYS } from "../types";

type Props = {
  status?: TuneStatus;
  lastRun: RunSummary | null;
  onApplied: () => void;
};

export function TuningPanel({ status, lastRun, onApplied }: Props) {
  const tune = useMutation({ mutationFn: () => tuneCb(lastRun) });
  const apply = useMutation({
    mutationFn: (s: CircuitBreakerSettings) => applyCbConfig(s),
    onSuccess: onApplied,
  });

  const result = tune.data;
  const clamps = new Map((result?.clamps ?? []).map((c) => [c.field, c]));

  return (
    <div className="card">
      <div className="card-head">
        <h3>LLM self-tune (advisory)</h3>
        {status && <span className="muted">model: {status.model}</span>}
      </div>

      {status && !status.configured && (
        <p className="warn">OPENAI_API_KEY is not set on the backend. Export it and restart to enable tuning.</p>
      )}

      <button
        className="primary"
        disabled={!status?.configured || tune.isPending}
        onClick={() => tune.mutate()}
      >
        {tune.isPending ? "Asking the model…" : "Call LLM to self-tune"}
      </button>
      {tune.isError && <p className="error">{(tune.error as Error).message}</p>}

      {result && (
        <>
          <table className="diff">
            <thead>
              <tr>
                <th>Knob</th>
                <th>Current</th>
                <th>Proposed</th>
                <th>Clamped (applied)</th>
              </tr>
            </thead>
            <tbody>
              {SETTING_KEYS.map((k) => {
                const c = clamps.get(k as string);
                const wasClamped = c?.wasClamped ?? false;
                return (
                  <tr key={k} className={wasClamped ? "clamped" : ""}>
                    <td>{k}</td>
                    <td>{String(result.current[k])}</td>
                    <td>{String(result.proposed[k])}</td>
                    <td>
                      {String(result.clamped[k])}
                      {wasClamped && <span className="badge">clamped</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div className="rationale">
            <strong>Model rationale</strong>
            <p>{result.rationale}</p>
          </div>

          <button className="apply" disabled={apply.isPending} onClick={() => apply.mutate(result.clamped)}>
            {apply.isPending ? "Applying…" : apply.isSuccess ? "Applied — captured the after run" : "Apply clamped config"}
          </button>
          {apply.isError && <p className="error">{(apply.error as Error).message}</p>}
        </>
      )}
    </div>
  );
}
