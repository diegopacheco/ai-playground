import { useEffect, useState } from "react";
import { fetchStats } from "./api";
import { AskPanel } from "./components/AskPanel";
import { Header } from "./components/Header";
import { SearchPanel } from "./components/SearchPanel";
import { SimilarPanel } from "./components/SimilarPanel";
import { Tabs } from "./components/Tabs";
import type { Hit, TabName } from "./types";

export function App() {
  const [tab, setTab] = useState<TabName>("ask");
  const [videos, setVideos] = useState<number | null>(null);
  const [selected, setSelected] = useState<Hit | null>(null);

  useEffect(() => {
    fetchStats()
      .then((stats) => setVideos(stats.videos))
      .catch(() => setVideos(null));
  }, []);

  function showSimilar(hit: Hit) {
    setSelected(hit);
    setTab("similar");
    window.scrollTo({ top: 0 });
  }

  return (
    <div className="page">
      <Header videos={videos} />
      <Tabs active={tab} onChange={setTab} />
      <main>
        <div hidden={tab !== "ask"}>
          <AskPanel onSimilar={showSimilar} />
        </div>
        <div hidden={tab !== "search"}>
          <SearchPanel onSimilar={showSimilar} />
        </div>
        <div hidden={tab !== "similar"}>
          <SimilarPanel video={selected} onSimilar={showSimilar} />
        </div>
      </main>
    </div>
  );
}
