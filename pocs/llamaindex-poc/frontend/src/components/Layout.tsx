import { useQuery } from "@tanstack/react-query";
import { Link, Outlet, useRouterState } from "@tanstack/react-router";
import { api } from "../api";
import { SparkPanel } from "./SparkPanel";

const TABS = [
  { to: "/", label: "1 · Ingest" },
  { to: "/chat", label: "2 · Chat (RAG)" },
  { to: "/rust", label: "3 · Rust llama" },
  { to: "/search", label: "4 · Search" },
  { to: "/agents", label: "5 · Agents" },
  { to: "/annotate", label: "6 · Annotate" },
] as const;

export function Layout() {
  const path = useRouterState({ select: (state) => state.location.pathname });
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15000 });
  const current = TABS.find((tab) => tab.to === path)?.label ?? path;

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <h1>LlamaIndex POC</h1>
          <p>PDF parsing, indexing and retrieval with LlamaIndex, local Ollama and Rust llama.cpp</p>
        </div>
        <div className="health">
          <span className={`pill ${health.data?.ollama.reachable ? "ok" : "err"}`}>
            ollama {health.data?.ollama.reachable ? "up" : "down"}
          </span>
          <span className="pill">{health.data?.index.embed_model ?? "…"}</span>
          <span className="pill">{health.data?.index.llm_model ?? "…"}</span>
          <span className="pill">{health.data?.index.documents ?? 0} docs</span>
          <span className="pill">{health.data?.index.chunks ?? 0} chunks</span>
        </div>
      </header>

      <nav className="tabs">
        {TABS.map((tab) => (
          <Link key={tab.to} to={tab.to} className={`tab ${path === tab.to ? "active" : ""}`}>
            {tab.label}
          </Link>
        ))}
      </nav>

      <Outlet />
      <SparkPanel context={`Screen: ${current}`} />
    </div>
  );
}
