import type { Chunk } from "../types";

type ChunksViewProps = { readonly chunks: readonly Chunk[] };

export function ChunksView({ chunks }: ChunksViewProps) {
  return (
    <ol className="chunks">
      {chunks.map((c) => (
        <li key={c.index} className="card">
          <div className="chunk-head">
            <span className="chunk-index">#{c.index + 1}</span>
            <span className="muted">{c.headings.length ? c.headings.join(" > ") : "no heading"}</span>
          </div>
          <p>{c.text}</p>
        </li>
      ))}
    </ol>
  );
}
