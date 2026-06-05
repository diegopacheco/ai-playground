import type { Label } from "../lib/types";

function readableText(hex: string): string {
  const clean = hex.replace("#", "");
  if (clean.length !== 6) {
    return "#0d1117";
  }
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance > 0.6 ? "#0d1117" : "#ffffff";
}

export function LabelChip({ label }: { label: Label }) {
  const color = label.color && label.color.length === 6 ? `#${label.color}` : "#8b949e";
  return (
    <span className="label-chip" style={{ background: color, color: readableText(label.color) }}>
      {label.name}
    </span>
  );
}

export function StateBadge({ state }: { state: string }) {
  const key = state.toLowerCase();
  return <span className={`state-badge state-${key}`}>{state}</span>;
}

export function TypeBadge({ type }: { type: string }) {
  return <span className={`type-badge type-${type.toLowerCase()}`}>{type}</span>;
}
