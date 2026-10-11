import { useState } from "react";
import { ask } from "../api";
import type { AskResponse, Filters } from "../types";
import { ProductList } from "./ProductList";
import { QueryForm } from "./QueryForm";

type Props = {
  filters: Filters;
};

export function AskPanel({ filters }: Props) {
  const [result, setResult] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function submit(question: string) {
    setLoading(true);
    setError("");
    ask(question, filters)
      .then(setResult)
      .catch((cause: Error) => setError(cause.message))
      .finally(() => setLoading(false));
  }

  return (
    <section>
      <QueryForm placeholder="What are you shopping for?" action="Ask" loading={loading} onSubmit={submit} />
      {loading && <p className="status">Retrieving products from Qdrant and asking llama3.2...</p>}
      {error && <p className="error">{error}</p>}
      {result && !loading && (
        <>
          <div className="card answer">{result.answer}</div>
          <h2>Sources ({result.sources.length})</h2>
          <ProductList hits={result.sources} />
        </>
      )}
    </section>
  );
}
