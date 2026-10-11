import { askBlog } from "../api";
import type { Answer, Filters } from "../types";
import { useRequest } from "../useRequest";
import { PostCard } from "./PostCard";
import { QueryForm } from "./QueryForm";

type Props = {
  filters: Filters;
};

const QUESTIONS = [
  "What does Diego think about microservices?",
  "How should a team deal with technical debt?",
  "What are the trade-offs of using Rust?",
];

export function AskPanel({ filters }: Props) {
  const request = useRequest<Answer>();

  return (
    <section>
      <QueryForm
        placeholder="Ask anything the blog has written about"
        action="Ask"
        loading={request.loading}
        suggestions={QUESTIONS}
        onSubmit={(question) => request.run(() => askBlog(question, filters))}
      />
      {request.error && <p className="error">{request.error}</p>}
      {request.loading && <p className="status">Retrieving excerpts from Qdrant and writing the answer…</p>}
      {request.data && !request.loading && (
        <>
          <article className="answer">
            <h2>Answer</h2>
            <p>{request.data.answer}</p>
          </article>
          <h2 className="section-title">Sources ({request.data.sources.length})</h2>
          <div className="cards">
            {request.data.sources.map((hit) => (
              <PostCard key={hit.url} hit={hit} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}
