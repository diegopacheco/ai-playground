import type { Stats } from "../types";

export function Header({ stats }: { stats: Stats | null }) {
  return (
    <header className="header">
      <div>
        <h1>Semantic Cache Q/A</h1>
        <p className="subtitle">Rust proxy in front of Claude Code with a Redis 8 vector cache</p>
      </div>
      {stats && (
        <div className="models">
          <span>Claude: {stats.claudeModel}</span>
          <span>Embeddings: {stats.embedModel}</span>
        </div>
      )}
    </header>
  );
}
