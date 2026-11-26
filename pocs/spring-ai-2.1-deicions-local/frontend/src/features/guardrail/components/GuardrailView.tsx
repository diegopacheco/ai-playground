import { Latency } from "../../../components/Latency";
import { Pill } from "../../../components/Pill";
import { ProbabilityBar } from "../../../components/ProbabilityBar";
import { decimal } from "../../../format";
import { outcomeInfo } from "../outcome";
import type { Screening } from "../types";

type GuardrailViewProps = { readonly result: Screening };

export function GuardrailView({ result }: GuardrailViewProps) {
  const info = outcomeInfo(result.outcome);
  const hit = new Set([...result.triggered, ...result.flagged]);
  return (
    <div className="result">
      <div className="verdict">
        <Pill text={result.outcome} tone={info.tone} />
        <p>{info.meaning}</p>
      </div>
      <article className="card wide">
        <h3>
          {result.battery.toLowerCase()} battery <span className="kind">severity {decimal(result.severity)} of 3</span>
        </h3>
        {Object.entries(result.scores).map(([hazard, p]) => (
          <ProbabilityBar key={hazard} label={hazard} value={p} chosen={hit.has(hazard)} />
        ))}
        <p className="mono sub">{result.summary}</p>
      </article>
      <Latency ms={result.latencyMs} />
    </div>
  );
}
