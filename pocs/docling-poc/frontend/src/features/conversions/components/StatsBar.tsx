import type { Stats } from "../types";

type StatsBarProps = { readonly stats: Stats };

const LABELS: readonly (readonly [keyof Stats, string])[] = [
  ["pages", "Pages"],
  ["texts", "Text items"],
  ["headings", "Headings"],
  ["tables", "Tables"],
  ["pictures", "Pictures"],
  ["words", "Words"],
];

export function StatsBar({ stats }: StatsBarProps) {
  return (
    <div className="stats">
      {LABELS.map(([key, label]) => (
        <div key={key} className="stat">
          <span className="stat-value">{stats[key]}</span>
          <span className="stat-label">{label}</span>
        </div>
      ))}
    </div>
  );
}
