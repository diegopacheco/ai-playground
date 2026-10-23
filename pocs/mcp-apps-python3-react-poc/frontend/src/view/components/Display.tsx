type DisplayProps = {
  expression: string;
  previous: string;
  error: string | null;
  busy: boolean;
};

export function Display({ expression, previous, error, busy }: DisplayProps) {
  return (
    <div className="display" role="status" aria-live="polite">
      <div className="display-previous">{error ? "" : previous}</div>
      <div className={error ? "display-main error" : "display-main"} data-testid="display">
        {error ?? (expression || "0")}
      </div>
      <div className="display-busy">{busy ? "calculating" : ""}</div>
    </div>
  );
}
