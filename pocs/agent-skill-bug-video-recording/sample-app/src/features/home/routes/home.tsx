export function HomePage() {
  return (
    <section className="page">
      <h1>Welcome</h1>
      <p className="lede">A small shop used to exercise the bug recorder.</p>
      <div className="tiles">
        <div className="tile">
          <span className="tile-k">Catalog</span>
          <span className="tile-v">4 products</span>
        </div>
        <div className="tile">
          <span className="tile-k">Cart</span>
          <span className="tile-v">2 items</span>
        </div>
        <div className="tile">
          <span className="tile-k">Dashboard</span>
          <span className="tile-v">3 metrics</span>
        </div>
      </div>
    </section>
  )
}
