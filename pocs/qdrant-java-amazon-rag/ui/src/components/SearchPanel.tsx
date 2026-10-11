import { useState } from "react";
import { search } from "../api";
import type { Filters, Hit } from "../types";
import { ProductList } from "./ProductList";
import { QueryForm } from "./QueryForm";

type Props = {
  filters: Filters;
};

export function SearchPanel({ filters }: Props) {
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function submit(query: string) {
    setLoading(true);
    setError("");
    search(query, filters)
      .then((response) => setHits(response.hits))
      .catch((cause: Error) => setError(cause.message))
      .finally(() => setLoading(false));
  }

  return (
    <section>
      <QueryForm placeholder="Describe a product" action="Search" loading={loading} onSubmit={submit} />
      {loading && <p className="status">Searching Qdrant...</p>}
      {error && <p className="error">{error}</p>}
      {hits && !loading && (
        <>
          <h2>{hits.length === 0 ? "No products match" : `Results (${hits.length})`}</h2>
          <ProductList hits={hits} />
        </>
      )}
    </section>
  );
}
