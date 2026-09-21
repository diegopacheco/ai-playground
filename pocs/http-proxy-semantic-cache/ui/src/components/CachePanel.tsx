import { useCallback, useState } from "react";
import { api } from "../api/client";
import { useRemote } from "../hooks/useRemote";
import { formatTime } from "../lib/format";

export function CachePanel({ version, onCleared }: { version: number; onCleared: () => void }) {
  const [refresh, setRefresh] = useState(0);
  const load = useCallback(() => api.entries(), []);
  const { data: entries, error } = useRemote(load, version + refresh);

  const clear = async () => {
    await api.clear();
    onCleared();
  };

  return (
    <section className="panel">
      <div className="toolbar">
        <span>{entries ? `${entries.length} cached answers` : "Loading..."}</span>
        <div className="actions">
          <button className="secondary" onClick={() => setRefresh((value) => value + 1)}>
            Refresh
          </button>
          <button className="danger" onClick={clear} disabled={!entries?.length}>
            Clear cache
          </button>
        </div>
      </div>
      {error && <p className="error">{error}</p>}
      {entries?.length === 0 && <p className="empty">The cache is empty. Ask a question to fill it.</p>}
      {!!entries?.length && (
        <table className="cache-table">
          <thead>
            <tr>
              <th>Question</th>
              <th>Answer</th>
              <th>Hits</th>
              <th>Cached at</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => (
              <tr key={entry.key}>
                <td className="question">{entry.question}</td>
                <td className="answer-cell">{entry.answer}</td>
                <td className="hits">{entry.hits}</td>
                <td className="time">{formatTime(entry.createdAt)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
