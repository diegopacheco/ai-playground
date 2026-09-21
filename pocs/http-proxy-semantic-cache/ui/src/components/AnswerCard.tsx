import { formatLatency, formatSimilarity } from "../lib/format";
import type { AskResult } from "../types";
import { CacheBadge } from "./CacheBadge";

export function AnswerCard({ result }: { result: AskResult }) {
  return (
    <article className={result.cached ? "card hit" : "card miss"}>
      <div className="card-head">
        <CacheBadge cached={result.cached} />
        <span className="meta">{formatLatency(result.latencyMs)}</span>
        <span className="meta">nearest similarity {formatSimilarity(result.similarity)}</span>
      </div>
      <h3>{result.question}</h3>
      {result.cached && result.matchedQuestion && <p className="matched">Matched cached question: "{result.matchedQuestion}"</p>}
      {!result.cached && <p className="matched">Sent to Claude Code and stored in Redis</p>}
      <p className="answer">{result.answer}</p>
    </article>
  );
}
