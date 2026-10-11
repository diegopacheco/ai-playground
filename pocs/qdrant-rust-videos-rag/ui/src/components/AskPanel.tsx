import { useState } from "react";
import type { FormEvent } from "react";
import { askQuestion, errorMessage } from "../api";
import type { Answer, Hit } from "../types";
import { VideoCard } from "./VideoCard";

type Props = {
  onSimilar: (hit: Hit) => void;
};

export function AskPanel({ onSimilar }: Props) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      setAnswer(await askQuestion(question));
    } catch (failure) {
      setAnswer(null);
      setError(errorMessage(failure));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section>
      <form className="card form" onSubmit={submit}>
        <textarea
          rows={3}
          value={question}
          placeholder="Ask anything about the channel videos"
          onChange={(event) => setQuestion(event.target.value)}
        />
        <button className="primary" disabled={loading || !question.trim()}>
          {loading ? "Thinking..." : "Ask"}
        </button>
      </form>
      {loading && <p className="status">Retrieving videos and generating the answer...</p>}
      {error && <p className="status error">{error}</p>}
      {answer && !loading && (
        <>
          <div className="card answer">
            <h2>Answer</h2>
            <p>{answer.answer}</p>
          </div>
          <h2 className="section-title">Sources</h2>
          <div className="results">
            {answer.sources.map((hit) => (
              <VideoCard key={hit.id} hit={hit} onSimilar={onSimilar} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}
