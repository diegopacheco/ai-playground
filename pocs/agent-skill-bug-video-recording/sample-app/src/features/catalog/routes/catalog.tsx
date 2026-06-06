import { useQuery } from '@tanstack/react-query'
import { fetchProducts } from '../api/products'
import { ProductCard } from '../components/ProductCard'

export function CatalogPage() {
  const { data, isPending } = useQuery({ queryKey: ['products'], queryFn: fetchProducts })
  if (isPending || !data) return <p className="page">Loading…</p>
  return (
    <section className="page">
      <h1>Catalog</h1>
      <div className="grid">
        {data.map(p => (
          <ProductCard key={p.id} product={p} />
        ))}
      </div>
    </section>
  )
}
