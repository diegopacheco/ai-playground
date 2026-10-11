import { useState } from "react";
import type { FormEvent } from "react";

type Props = {
  placeholder: string;
  action: string;
  loading: boolean;
  onSubmit: (text: string) => void;
};

export function QueryForm({ placeholder, action, loading, onSubmit }: Props) {
  const [text, setText] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();
    onSubmit(text.trim());
  }

  return (
    <form className="query" onSubmit={submit}>
      <input value={text} placeholder={placeholder} onChange={(event) => setText(event.target.value)} />
      <button type="submit" disabled={loading || text.trim() === ""}>
        {loading ? "Working..." : action}
      </button>
    </form>
  );
}
