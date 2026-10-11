import { useState } from "react";
import type { Hit } from "../types";

type Props = {
  hit: Hit;
};

export function ProductCard({ hit }: Props) {
  const [imageBroken, setImageBroken] = useState(false);

  return (
    <article className="card product">
      {hit.image && !imageBroken && (
        <img src={hit.image} alt="" loading="lazy" referrerPolicy="no-referrer" onError={() => setImageBroken(true)} />
      )}
      <div className="product-body">
        <a className="product-name" href={hit.url} target="_blank" rel="noreferrer">
          {hit.name}
        </a>
        <div className="product-meta">
          <span className="tag">{hit.category}</span>
          <span className="price">{hit.price === null ? "No price" : `$${hit.price.toFixed(2)}`}</span>
          <span className="score">score {hit.score.toFixed(3)}</span>
        </div>
        <p className="about">{hit.about}</p>
      </div>
    </article>
  );
}
