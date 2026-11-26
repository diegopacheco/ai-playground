import { percent } from "../format";

type ProbabilityBarProps = { readonly label: string; readonly value: number; readonly chosen?: boolean };

export function ProbabilityBar({ label, value, chosen = false }: ProbabilityBarProps) {
  return (
    <div className={chosen ? "bar bar-chosen" : "bar"}>
      <span className="bar-label">{label}</span>
      <span className="bar-track">
        <span className="bar-fill" style={{ width: percent(value) }} />
      </span>
      <span className="bar-value">{percent(value)}</span>
    </div>
  );
}
