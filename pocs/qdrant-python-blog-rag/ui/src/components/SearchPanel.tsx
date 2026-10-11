import { searchPosts } from "../api";
import type { Filters, SearchResult } from "../types";
import { useRequest } from "../useRequest";
import { PostCard } from "./PostCard";
import { QueryForm } from "./QueryForm";

type Props = {
  filters: Filters;
};

const QUERIES = ["distributed systems trade-offs", "engineering culture", "AI coding agents"];

export function SearchPanel({ filters }: Props) {
  const request = useRequest<SearchResult>();

  return (
    <section>
      <QueryForm
        placeholder="Search posts by meaning, not by keyword"
        action="Search"
        loading={request.loading}
        suggestions={QUERIES}
        onSubmit={(query) => request.run(() => searchPosts(query, filters))}
      />
      {request.error && <p className="error">{request.error}</p>}
      {request.data && !request.loading && (
        <>
          <h2 className="section-title">Posts ({request.data.hits.length})</h2>
          {request.data.hits.length === 0 && <p className="status">No post matched these filters.</p>}
          <div className="cards">
            {request.data.hits.map((hit) => (
              <PostCard key={hit.url} hit={hit} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}
