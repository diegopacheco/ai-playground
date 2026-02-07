import { useState, useMemo } from "react";
import { getSolutions, deleteSolution, loadSolutionForPreview } from "../store/wizard";
import type { SavedSolution } from "../types";
import styles from "./SearchStep.module.css";

export function SearchStep() {
  const [query, setQuery] = useState("");
  const [solutions, setSolutions] = useState<SavedSolution[]>(getSolutions());

  const filtered = useMemo(() => {
    if (!query.trim()) return solutions;
    const lower = query.toLowerCase();
    return solutions.filter(
      (s) =>
        s.prompt.toLowerCase().includes(lower) ||
        s.style.toLowerCase().includes(lower)
    );
  }, [query, solutions]);

  function handleDelete(id: string) {
    deleteSolution(id);
    setSolutions(getSolutions());
  }

  function handleLoad(html: string) {
    loadSolutionForPreview(html);
  }

  function formatDate(ts: number): string {
    return new Date(ts).toLocaleString();
  }

  return (
    <div className={styles.container}>
      <input
        className={styles.searchInput}
        type="text"
        placeholder="Search solutions by prompt or style..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <div className={styles.resultCount}>
        {filtered.length} solution{filtered.length !== 1 ? "s" : ""} found
      </div>

      {filtered.length === 0 ? (
        <div className={styles.empty}>
          {solutions.length === 0
            ? "No solutions yet. Generate your first app!"
            : "No solutions match your search."}
        </div>
      ) : (
        <div className={styles.list}>
          {filtered.map((s) => (
            <div key={s.id} className={styles.card}>
              <div className={styles.cardInfo}>
                <div className={styles.cardPrompt}>{s.prompt}</div>
                <div className={styles.cardMeta}>
                  <span className={styles.styleBadge}>{s.style}</span>
                  <span>{formatDate(s.createdAt)}</span>
                </div>
              </div>
              <div className={styles.cardActions}>
                <button className={styles.loadBtn} onClick={() => handleLoad(s.html)}>
                  Preview
                </button>
                <button className={styles.deleteBtn} onClick={() => handleDelete(s.id)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
