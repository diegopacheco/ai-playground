import { useState } from "react";
import type { FormEvent } from "react";
import { errorMessage, searchVideos } from "../api";
import type { Hit } from "../types";
import { VideoCard } from "./VideoCard";

type Props = {
  onSimilar: (hit: Hit) => void;
};

function minutes(value: string): number | undefined {
  return value.trim() === "" ? undefined : Number(value);
}

export function SearchPanel({ onSimilar }: Props) {
  const [query, setQuery] = useState("");
  const [min, setMin] = useState("");
  const [max, setMax] = useState("");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      setHits(await searchVideos({ query, min_minutes: minutes(min), max_minutes: minutes(max) }));
    } catch (failure) {
      setHits(null);
      setError(errorMessage(failure));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section>
      <form className="card form" onSubmit={submit}>
        <input
          type="text"
          value={query}
          placeholder="Search videos by meaning"
          onChange={(event) => setQuery(event.target.value)}
        />
        <div className="form-row">
          <label>
            Min minutes
            <input type="number" min="0" value={min} onChange={(event) => setMin(event.target.value)} />
          </label>
          <label>
            Max minutes
            <input type="number" min="0" value={max} onChange={(event) => setMax(event.target.value)} />
          </label>
          <button className="primary" disabled={loading || !query.trim()}>
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </form>
      {loading && <p className="status">Searching Qdrant...</p>}
      {error && <p className="status error">{error}</p>}
      {hits && !loading && hits.length === 0 && <p className="status">No videos matched.</p>}
      {hits && !loading && (
        <div className="results">
          {hits.map((hit) => (
            <VideoCard key={hit.id} hit={hit} onSimilar={onSimilar} />
          ))}
        </div>
      )}
    </section>
  );
}
