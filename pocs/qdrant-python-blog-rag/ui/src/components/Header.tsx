import type { Stats } from "../types";

type Props = {
  stats: Stats | null;
  error: string;
};

function statsLabel(stats: Stats | null, error: string): string {
  if (error) {
    return "api unreachable";
  }
  if (!stats) {
    return "loading";
  }
  return `${stats.posts.toLocaleString()} posts · ${stats.chunks.toLocaleString()} chunks`;
}

export function Header({ stats, error }: Props) {
  return (
    <header className="header">
      <div>
        <h1>Blog RAG</h1>
        <p className="subtitle">Python 3.14 + Qdrant over every post of diego-pacheco.blogspot.com</p>
      </div>
      <span className={error ? "pill pill-error" : "pill"}>{statsLabel(stats, error)}</span>
    </header>
  );
}
