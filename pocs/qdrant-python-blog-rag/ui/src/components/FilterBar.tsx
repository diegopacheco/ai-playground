import type { Facet, Filters, Stats } from "../types";

type Props = {
  stats: Stats | null;
  filters: Filters;
  onChange: (filters: Filters) => void;
};

function options(facets: Facet[]) {
  return facets.map((facet) => (
    <option key={facet.name} value={facet.name}>
      {facet.name} ({facet.count})
    </option>
  ));
}

export function FilterBar({ stats, filters, onChange }: Props) {
  return (
    <section className="filters">
      <label>
        Label
        <select value={filters.label} onChange={(event) => onChange({ ...filters, label: event.target.value })}>
          <option value="">All labels</option>
          {options(stats?.labels ?? [])}
        </select>
      </label>
      <label>
        Year
        <select value={filters.year} onChange={(event) => onChange({ ...filters, year: event.target.value })}>
          <option value="">All years</option>
          {options(stats?.years ?? [])}
        </select>
      </label>
      <span className="hint">Filters run inside Qdrant as payload conditions, counts are indexed chunks.</span>
    </section>
  );
}
