import { useState } from "react";
import type { ConvertedDocument } from "../types";
import { ChunksView } from "./ChunksView";
import { TablesView } from "./TablesView";

const TABS = ["markdown", "preview", "tables", "chunks"] as const;
type Tab = (typeof TABS)[number];

const TITLES: Readonly<Record<Tab, string>> = {
  markdown: "Markdown",
  preview: "HTML preview",
  tables: "Tables",
  chunks: "RAG chunks",
};

type DocumentTabsProps = { readonly document: ConvertedDocument };

function TabBody({ tab, document }: { readonly tab: Tab; readonly document: ConvertedDocument }) {
  switch (tab) {
    case "markdown":
      return <pre className="code">{document.markdown}</pre>;
    case "preview":
      return <iframe className="preview" title="html preview" sandbox="" srcDoc={document.html} />;
    case "tables":
      return <TablesView tables={document.tables} />;
    case "chunks":
      return <ChunksView chunks={document.chunks} />;
    default: {
      const unreachable: never = tab;
      return unreachable;
    }
  }
}

export function DocumentTabs({ document }: DocumentTabsProps) {
  const [tab, setTab] = useState<Tab>("markdown");
  const counts: Readonly<Record<Tab, number | undefined>> = {
    markdown: undefined,
    preview: undefined,
    tables: document.tables.length,
    chunks: document.chunks.length,
  };
  return (
    <section>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t} type="button" className={t === tab ? "tab active" : "tab"} onClick={() => setTab(t)}>
            {TITLES[t]}
            {counts[t] !== undefined && <span className="count">{counts[t]}</span>}
          </button>
        ))}
      </nav>
      <TabBody tab={tab} document={document} />
    </section>
  );
}
