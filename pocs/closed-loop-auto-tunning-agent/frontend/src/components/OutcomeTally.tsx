const ORDER = ["SUCCESS", "FAILURE", "SHORT_CIRCUITED", "RATE_LIMITED", "REJECTED"];
const LABELS: Record<string, string> = {
  SUCCESS: "Success",
  FAILURE: "Failure",
  SHORT_CIRCUITED: "Short-circuited",
  RATE_LIMITED: "Rate-limited",
  REJECTED: "Rejected",
};

type Props = { tally: Record<string, number> };

export function OutcomeTally({ tally }: Props) {
  const shown = ORDER.filter((k) => k === "SUCCESS" || k === "FAILURE" || (tally[k] || 0) > 0);
  return (
    <div className="tally">
      {shown.map((k) => (
        <span key={k} className={`pill ${k.toLowerCase()}`}>
          <strong>{tally[k] || 0}</strong> {LABELS[k]}
        </span>
      ))}
    </div>
  );
}
