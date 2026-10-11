type Props = {
  videos: number | null;
};

export function Header({ videos }: Props) {
  return (
    <header className="header">
      <div>
        <h1>Videos RAG</h1>
        <p className="subtitle">Rust + Qdrant</p>
      </div>
      <span className="pill">{videos === null ? "index unavailable" : `${videos} videos indexed`}</span>
    </header>
  );
}
