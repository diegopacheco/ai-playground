type Props = {
  products: number | null;
};

export function Header({ products }: Props) {
  return (
    <header className="header">
      <div>
        <h1>Amazon Products RAG</h1>
        <p className="subtitle">Java 25 + Spring Boot + Qdrant</p>
      </div>
      <span className="pill">{products === null ? "loading" : `${products.toLocaleString()} products indexed`}</span>
    </header>
  );
}
