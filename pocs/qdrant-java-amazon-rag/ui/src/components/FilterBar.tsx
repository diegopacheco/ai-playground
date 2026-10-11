import type { Category, Filters } from "../types";

type Props = {
  categories: Category[];
  filters: Filters;
  onChange: (filters: Filters) => void;
};

export function FilterBar({ categories, filters, onChange }: Props) {
  return (
    <section className="card filters">
      <label>
        Category
        <select value={filters.category} onChange={(event) => onChange({ ...filters, category: event.target.value })}>
          <option value="">All categories</option>
          {categories.map((category) => (
            <option key={category.name} value={category.name}>
              {category.name} ({category.count})
            </option>
          ))}
        </select>
      </label>
      <label>
        Min price
        <input
          type="number"
          min="0"
          placeholder="0"
          value={filters.minPrice}
          onChange={(event) => onChange({ ...filters, minPrice: event.target.value })}
        />
      </label>
      <label>
        Max price
        <input
          type="number"
          min="0"
          placeholder="any"
          value={filters.maxPrice}
          onChange={(event) => onChange({ ...filters, maxPrice: event.target.value })}
        />
      </label>
    </section>
  );
}
