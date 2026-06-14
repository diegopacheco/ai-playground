import { memo } from "react";
import PriceTag from "./PriceTag.jsx";
import StatBadge from "./StatBadge.jsx";
import { products } from "../data.js";

function ProductCard({ product = products[0] }) {
  return (
    <article className="product-card">
      <header className="product-card__head">
        <h3 className="product-card__name">{product.name}</h3>
        <span className="product-card__cat">{product.category}</span>
      </header>
      <PriceTag amount={product.price} />
      <div className="product-card__stats">
        <StatBadge label="Stock" value={product.stock} delta={product.trend} />
        <StatBadge label="Rating" value={product.rating} unit="★" delta={0} />
      </div>
    </article>
  );
}

export default memo(ProductCard);
