import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";
import { DocumentPicker } from "../components/DocumentPicker";
import { Sources } from "../components/Sources";
import type { SearchResponse } from "../types";

export function Search() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("semantic");
  const [topK, setTopK] = useState(10);
  const [docIds, setDocIds] = useState<string[]>([]);

  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });

  const run = useMutation<SearchResponse, Error, void>({
    mutationFn: () => api.search({ query, mode, top_k: topK, doc_ids: docIds }),
  });

  return (
    <>
      <div className="card">
        <h2>Search across every indexed file</h2>
        <p className="hint">
          Semantic search embeds the query and ranks chunks by cosine similarity. Keyword search
          scans the docstore for literal occurrences and ranks by hit count.
        </p>
        <div className="row">
          <input
            type="text"
            className="grow"
            value={query}
            placeholder="rollback procedure"
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && query.trim()) run.mutate();
            }}
          />
          <select value={mode} style={{ width: 130 }} onChange={(event) => setMode(event.target.value)}>
            <option value="semantic">semantic</option>
            <option value="keyword">keyword</option>
          </select>
          <input
            type="number"
            min={1}
            max={50}
            value={topK}
            style={{ width: 80 }}
            onChange={(event) => setTopK(Number(event.target.value))}
          />
          <button className="primary" disabled={!query.trim() || run.isPending} onClick={() => run.mutate()}>
            {run.isPending ? "searching…" : "Search"}
          </button>
        </div>
        <div style={{ marginTop: 10 }}>
          <DocumentPicker
            documents={documents.data?.documents ?? []}
            selected={docIds}
            onChange={setDocIds}
          />
        </div>
      </div>

      <div className="card">
        {run.isError && <div className="error">{run.error.message}</div>}
        {run.data && (
          <p className="hint">
            {run.data.hits.length} hits · {run.data.mode} · {run.data.elapsed_seconds}s
          </p>
        )}
        {run.data?.hits.length === 0 && <p className="muted">no matches</p>}
        {run.data && <Sources sources={run.data.hits} />}
        {!run.data && !run.isError && <p className="muted">Run a search to see matching chunks.</p>}
      </div>
    </>
  );
}
