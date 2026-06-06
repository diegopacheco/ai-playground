import type { Product } from '../api/products'

export function ProductCard({ product }: { product: Product }) {
  return (
    <article className="card">
      <div className="card-media" />
      <h3 className="card-title">{product.name}</h3>
      <div className="card-foot">
        <span className="card-price">${product.price}</span>
        <button className="card-add">Add</button>
      </div>
    </article>
  )
}
