import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { fetchRun } from "../api";
import { percent, scoreTone, shortTime, stringify } from "../format";

export function RunDetail({ runId }: { runId: string }) {
  const run = useQuery({ queryKey: ["run", runId], queryFn: () => fetchRun(runId) });
  if (run.isPending) return <p className="muted">Loading run</p>;
  if (run.isError) return <div className="alert">{run.error.message}</div>;
  const r = run.data;
  return (
    <>
      <Link to="/runs" className="back">
        Back to runs
      </Link>
      <section className="run-head">
        <div>
          <span className="tenant">{r.tenant}</span>
          <h2>
            {r.suite} <span className="mono muted">#{r.id}</span>
          </h2>
          <p className="muted">{shortTime(r.created_at)}</p>
        </div>
        <div className={`big-score ${scoreTone(r.score)}`}>
          {percent(r.score)}
          <span>
            {r.passed} of {r.passed + r.failed} cases passed
          </span>
        </div>
      </section>
      {r.cases.map((c) => (
        <section key={c.name} className={`case ${c.passed ? "ok" : "ko"}`}>
          <div className="case-head">
            <b>{c.name}</b>
            <span className={c.passed ? "pass" : "fail"}>{c.passed ? "PASS" : "FAIL"}</span>
            <span className="muted mono">{c.latency_ms} ms</span>
          </div>
          <div className="io">
            <div>
              <span className="label">Input</span>
              <pre>{c.input || "-"}</pre>
            </div>
            <div>
              <span className="label">Output</span>
              <pre>{c.output || "-"}</pre>
            </div>
          </div>
          <ul className="checks">
            {c.checks.map((check, i) => (
              <li key={i} className={check.passed ? "pass" : "fail"}>
                <span className="mono">{check.passed ? "ok " : "x  "}</span>
                <b>{check.type}</b> <span className="mono muted">{stringify(check.expected)}</span>
                <span className="detail">{check.detail}</span>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </>
  );
}
