import { useEffect, useState } from "react";
import { fetchStats } from "./api";
import { AskPanel } from "./components/AskPanel";
import { FilterBar } from "./components/FilterBar";
import { Header } from "./components/Header";
import { SearchPanel } from "./components/SearchPanel";
import { Tabs } from "./components/Tabs";
import type { Filters, Stats, TabName } from "./types";

export function App() {
  const [tab, setTab] = useState<TabName>("ask");
  const [stats, setStats] = useState<Stats | null>(null);
  const [statsError, setStatsError] = useState("");
  const [filters, setFilters] = useState<Filters>({ label: "", year: "" });

  useEffect(() => {
    fetchStats()
      .then(setStats)
      .catch((failure: Error) => setStatsError(failure.message));
  }, []);

  return (
    <div className="page">
      <Header stats={stats} error={statsError} />
      <Tabs active={tab} onChange={setTab} />
      <FilterBar stats={stats} filters={filters} onChange={setFilters} />
      <main>
        <div hidden={tab !== "ask"}>
          <AskPanel filters={filters} />
        </div>
        <div hidden={tab !== "search"}>
          <SearchPanel filters={filters} />
        </div>
      </main>
    </div>
  );
}
