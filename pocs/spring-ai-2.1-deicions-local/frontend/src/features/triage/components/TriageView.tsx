import { ConfidenceLine } from "../../../components/ConfidenceLine";
import { Latency } from "../../../components/Latency";
import { Pill } from "../../../components/Pill";
import { ProbabilityBar } from "../../../components/ProbabilityBar";
import { decimal } from "../../../format";
import { defectZones, routeInfo } from "../route";
import type { Triage } from "../types";

type TriageViewProps = { readonly result: Triage };

export function TriageView({ result }: TriageViewProps) {
  const route = routeInfo(result.route);
  return (
    <div className="result">
      <div className="verdict">
        <Pill text={route.label} tone={route.tone} />
        <p>{route.action}</p>
      </div>
      <div className="cards">
        <article className="card">
          <h3>defect <span className="kind">YesNoQuestion</span></h3>
          <p className="answer">{result.defect.yes ? "yes" : "no"}</p>
          <ConfidenceLine
            value={result.defect.probability}
            zones={defectZones(result.thresholds)}
            caption="Probability of yes. The boolean hides how sure the model is; the line does not."
          />
        </article>
        <article className="card">
          <h3>topic <span className="kind">ChoiceQuestion</span></h3>
          <p className="answer">{result.topic.value}</p>
          {Object.entries(result.topic.probabilities).map(([option, p]) => (
            <ProbabilityBar key={option} label={option} value={p} chosen={option === result.topic.value} />
          ))}
        </article>
        <article className="card">
          <h3>satisfaction <span className="kind">ScoreQuestion</span></h3>
          <p className="answer">
            {result.satisfaction.label} <span className="sub">expected level {decimal(result.satisfaction.expectedLevel)}</span>
          </p>
          {result.satisfaction.levels.map((level, i) => (
            <ProbabilityBar key={level} label={level} value={result.satisfaction.probabilities[i] ?? 0} chosen={i === result.satisfaction.nearestLevel} />
          ))}
        </article>
      </div>
      <Latency ms={result.latencyMs} model={result.model} />
    </div>
  );
}
