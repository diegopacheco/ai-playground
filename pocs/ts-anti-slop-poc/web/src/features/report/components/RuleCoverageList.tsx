import type { RuleCoverage } from "../types.ts";

type RuleCoverageListProps = { readonly rules: readonly RuleCoverage[] };

export function RuleCoverageList({ rules }: RuleCoverageListProps) {
  const max = Math.max(1, ...rules.map((r) => r.hits));

  return (
    <section className="card">
      <h2>Rules that fired</h2>
      <ul className="bars">
        {rules.map((r) => (
          <li key={r.rule}>
            <code>{r.rule}</code>
            <span className="bar" style={{ width: `${(r.hits / max) * 100}%` }} />
            <span className="hits">{r.hits}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
