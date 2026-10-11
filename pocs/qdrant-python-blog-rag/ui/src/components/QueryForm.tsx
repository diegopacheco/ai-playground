import { useState } from "react";
import type { FormEvent } from "react";

type Props = {
  placeholder: string;
  action: string;
  loading: boolean;
  suggestions: string[];
  onSubmit: (text: string) => void;
};

export function QueryForm({ placeholder, action, loading, suggestions, onSubmit }: Props) {
  const [text, setText] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();
    if (text.trim()) {
      onSubmit(text.trim());
    }
  }

  function pick(suggestion: string) {
    setText(suggestion);
    onSubmit(suggestion);
  }

  return (
    <div className="query">
      <form className="query-form" onSubmit={submit}>
        <input value={text} placeholder={placeholder} onChange={(event) => setText(event.target.value)} />
        <button className="primary" disabled={loading || !text.trim()}>
          {loading ? "Working…" : action}
        </button>
      </form>
      <div className="suggestions">
        {suggestions.map((suggestion) => (
          <button key={suggestion} className="chip" disabled={loading} onClick={() => pick(suggestion)}>
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}
