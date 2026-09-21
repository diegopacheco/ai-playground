import { useCallback, useState } from "react";
import { api } from "./api/client";
import { AskPanel } from "./components/AskPanel";
import { CachePanel } from "./components/CachePanel";
import { Header } from "./components/Header";
import { StatsBar } from "./components/StatsBar";
import { Tabs, type Tab } from "./components/Tabs";
import { useRemote } from "./hooks/useRemote";

type TabKey = "ask" | "cache";

const tabs: Tab<TabKey>[] = [
  { key: "ask", label: "Ask" },
  { key: "cache", label: "Cache" },
];

export function App() {
  const [tab, setTab] = useState<TabKey>("ask");
  const [version, setVersion] = useState(0);
  const changed = useCallback(() => setVersion((value) => value + 1), []);
  const loadStats = useCallback(() => api.stats(), []);
  const { data: stats, error } = useRemote(loadStats, version);

  return (
    <main className="app">
      <Header stats={stats} />
      <StatsBar stats={stats} />
      {error && <p className="error">Proxy unreachable: {error}</p>}
      <Tabs tabs={tabs} active={tab} onChange={setTab} />
      <div hidden={tab !== "ask"}>
        <AskPanel onAnswered={changed} />
      </div>
      {tab === "cache" && <CachePanel version={version} onCleared={changed} />}
    </main>
  );
}
