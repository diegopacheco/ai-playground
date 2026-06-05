import { useState } from "react";
import { Link, Outlet } from "@tanstack/react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./lib/api";
import type { SyncResponse } from "./lib/types";

const tabs = [
  { to: "/", label: "Repos" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/issues", label: "Issues" },
  { to: "/action-center", label: "Action Center" },
  { to: "/insights", label: "Insights" },
  { to: "/settings", label: "Settings" },
];

export function App() {
  const queryClient = useQueryClient();
  const [result, setResult] = useState<SyncResponse | null>(null);

  const sync = useMutation({
    mutationFn: api.sync,
    onSuccess: (data) => {
      setResult(data);
      void queryClient.invalidateQueries();
    },
  });

  const failures = result?.results.filter((r) => r.status !== "ok").length ?? 0;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◉</span>
          <span className="brand-name">GitHub Control Panel</span>
        </div>
        <div className="sync-area">
          {sync.isError && <span className="sync-status err">sync failed</span>}
          {result && !sync.isPending && (
            <span className={`sync-status ${failures ? "warn" : "ok"}`}>
              synced {result.repos} repo{result.repos === 1 ? "" : "s"}
              {failures ? ` · ${failures} failed` : ""}
            </span>
          )}
          <button className="sync-btn" disabled={sync.isPending} onClick={() => sync.mutate()}>
            {sync.isPending ? "Syncing…" : "Sync"}
          </button>
        </div>
      </header>

      <nav className="tabbar">
        {tabs.map((tab) => (
          <Link
            key={tab.to}
            to={tab.to}
            className="tab"
            activeOptions={{ exact: tab.to === "/" }}
            activeProps={{ className: "tab active" }}
          >
            {tab.label}
          </Link>
        ))}
      </nav>

      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
