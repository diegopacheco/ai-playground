import { useQuery } from '@tanstack/react-query';
import { getCatalog, money, type Product } from '../api';

function Card({ p }: { p: Product }) {
  return (
    <article className="card" data-sku={p.sku} data-price={p.priceCents} style={{ ['--accent' as string]: p.accent }}>
      <div className="card-band" />
      <h2>{p.name}</h2>
      <p className="muted">{p.tagline}</p>
      <div className="card-foot">
        <span className="price">{money(p.priceCents, p.currency)}</span>
        <span className="stock">{p.stock} in stock</span>
      </div>
    </article>
  );
}

export function Catalog() {
  const { data, isLoading } = useQuery({ queryKey: ['catalog'], queryFn: getCatalog });
  return (
    <section>
      <div className="page-head">
        <h1>Catalog</h1>
        <p className="muted">Products an autonomous buyer can browse and pay for over a signed TAP handshake.</p>
      </div>
      {isLoading ? (
        <p className="muted">loading catalog...</p>
      ) : (
        <div className="grid">
          {data?.map((p) => <Card key={p.sku} p={p} />)}
        </div>
      )}
    </section>
  );
}
