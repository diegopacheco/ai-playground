import { useAsk } from "../hooks/useAsk";
import { AnswerCard } from "./AnswerCard";
import { AskForm } from "./AskForm";

export function AskPanel({ onAnswered }: { onAnswered: () => void }) {
  const { history, error, submit, pending } = useAsk(onAnswered);
  return (
    <section className="panel">
      <AskForm submit={submit} pending={pending} />
      {pending && <p className="status">Embedding the question and searching the cache...</p>}
      {error && <p className="error">{error}</p>}
      {history.length === 0 && !pending && <p className="empty">No questions yet in this session.</p>}
      <div className="cards">
        {history.map((result, index) => (
          <AnswerCard key={`${history.length - index}`} result={result} />
        ))}
      </div>
    </section>
  );
}
