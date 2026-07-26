import type { Source } from "../types";

export function Sources({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <div>
      {sources.map((source) => (
        <div className="source" key={`${source.doc_id}-${source.position}`}>
          <div className="meta">
            [{source.position}] {source.file_name} · page {source.page} · score {source.score}
          </div>
          <div className="body">{source.text}</div>
        </div>
      ))}
    </div>
  );
}
