import { useEffect, useState } from "react";
import { fetchCategories, fetchStats } from "./api";
import { AskPanel } from "./components/AskPanel";
import { FilterBar } from "./components/FilterBar";
import { Header } from "./components/Header";
import { SearchPanel } from "./components/SearchPanel";
import { Tabs } from "./components/Tabs";
import type { Category, Filters, Tab } from "./types";

export function App() {
  const [products, setProducts] = useState<number | null>(null);
  const [categories, setCategories] = useState<Category[]>([]);
  const [filters, setFilters] = useState<Filters>({ category: "", minPrice: "", maxPrice: "" });
  const [tab, setTab] = useState<Tab>("ask");
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([fetchStats(), fetchCategories()])
      .then(([stats, list]) => {
        setProducts(stats.products);
        setCategories(list);
      })
      .catch((cause: Error) => setError(cause.message));
  }, []);

  return (
    <div className="page">
      <Header products={products} />
      {error && <p className="error">Could not load the catalog: {error}</p>}
      <FilterBar categories={categories} filters={filters} onChange={setFilters} />
      <Tabs active={tab} onChange={setTab} />
      <div hidden={tab !== "ask"}>
        <AskPanel filters={filters} />
      </div>
      <div hidden={tab !== "search"}>
        <SearchPanel filters={filters} />
      </div>
    </div>
  );
}
