import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { score } from "../scorecard";
import type { ScenarioResult } from "../types";

function ScenarioCard({ r }: { r: ScenarioResult }) {
  const tone = r.intent === "legit" ? (r.verdict === "ALLOWED" ? "good" : "warn") : r.secure ? "good" : "bad";
  return (
    <div className={`scenario ${tone}`}>
      <div className="scenario-head">
        <span className={`badge ${r.verdict.toLowerCase()}`}>{r.verdict}</span>
        <span className="layer">{r.layer}</span>
        <span className="muted">{r.intent === "attack" ? "attack" : "legit need"}</span>
      </div>
      <h3>{r.title}</h3>
      <code className="cmd">$ {r.command}</code>
      <pre>{r.evidence || `(exit ${r.exitCode}, no output)`}</pre>
      {r.intent === "legit" && r.verdict === "BLOCKED" && <p className="tip">Default deny. Add a rule in the Policy tab to allow it.</p>}
    </div>
  );
}

export function AttackLab({ name }: { name: string }) {
  const client = useQueryClient();
  const run = useMutation({
    mutationFn: () => api.attacks(name),
    onSettled: () => client.invalidateQueries({ queryKey: ["events", name] }),
  });
  const results = run.data ?? [];
  const s = score(results);
  return (
    <section className="card">
      <div className="lab-head">
        <div>
          <h2>Attack Lab</h2>
          <p className="hint">Plays a misbehaving agent: data exfiltration, cloud credential theft, tampering, persistence, namespace escape. Every attack runs for real inside the sandbox, in parallel.</p>
        </div>
        <button type="button" onClick={() => run.mutate()} disabled={run.isPending}>
          {run.isPending ? "Attacking..." : results.length ? "Run again" : "Launch attacks"}
        </button>
      </div>
      {run.error && <em className="error">{run.error.message}</em>}
      {results.length > 0 && (
        <div className="score">
          <div className={`grade grade-${s.grade}`}>{s.grade}</div>
          <div>
            <strong>{s.blocked} / {s.attacks} attacks blocked</strong>
            <div className="muted">{s.leaks.length === 0 ? "Nothing got out." : `Leaked: ${s.leaks.map((l) => l.title).join(", ")}`} {s.legitAllowed} of {s.legit} legit reads allowed by your rules.</div>
          </div>
        </div>
      )}
      <div className="scenarios">
        {results.map((r) => <ScenarioCard key={r.id} r={r} />)}
      </div>
    </section>
  );
}
