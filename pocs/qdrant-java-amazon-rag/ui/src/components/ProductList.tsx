import type { Hit } from "../types";
import { ProductCard } from "./ProductCard";

type Props = {
  hits: Hit[];
};

export function ProductList({ hits }: Props) {
  return (
    <div className="products">
      {hits.map((hit) => (
        <ProductCard key={hit.id} hit={hit} />
      ))}
    </div>
  );
}
