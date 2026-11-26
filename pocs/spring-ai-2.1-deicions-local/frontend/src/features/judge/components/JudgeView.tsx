import { Latency } from "../../../components/Latency";
import { Pill } from "../../../components/Pill";
import { decimal } from "../../../format";
import { findingTone } from "../outcome";
import type { Verdict } from "../types";

type JudgeViewProps = { readonly result: Verdict };

export function JudgeView({ result }: JudgeViewProps) {
  return (
    <div className="result">
      <div className="verdict">
        <Pill text={result.passed ? "PASSED" : "FAILED"} tone={result.passed ? "good" : "bad"} />
        <p>{result.passed ? "Every criterion met its threshold." : "At least one criterion missed its threshold."}</p>
      </div>
      <div className="cards">
        {result.findings.map((f) => (
          <article key={f.name} className="card">
            <h3>
              {f.name} <Pill text={f.outcome} tone={findingTone(f.outcome)} />
            </h3>
            <p className="answer mono">{decimal(f.value)}</p>
            <p className="sub">
              {f.measure}, threshold {decimal(f.threshold)}
            </p>
            {f.detail ? <p className="sub">{f.detail}</p> : null}
          </article>
        ))}
      </div>
      {result.feedback ? (
        <article className="card wide feedback">
          <h3>Feedback for a second try</h3>
          <pre>{result.feedback}</pre>
        </article>
      ) : null}
      <Latency ms={result.latencyMs} />
    </div>
  );
}
