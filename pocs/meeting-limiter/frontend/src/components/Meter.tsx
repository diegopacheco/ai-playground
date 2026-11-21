type Tone = "healthy" | "low" | "empty" | "neutral";

type Props = { readonly value: number; readonly max: number; readonly tone: Tone; readonly label: string };

export function Meter({ value, max, tone, label }: Props) {
  const pct = max <= 0 ? 0 : Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <span className={`meter meter--${tone}`} role="meter" aria-valuemin={0} aria-valuemax={max} aria-valuenow={value} aria-label={label}>
      <span style={{ width: `${pct}%` }} />
    </span>
  );
}
