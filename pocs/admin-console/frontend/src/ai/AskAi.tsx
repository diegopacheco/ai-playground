import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { createPortal } from "react-dom";
import { Badge } from "@design/Badge/Badge";
import { Button } from "@design/Button/Button";
import { api } from "@lib/api";
import type { AiSuggestion, ApiError } from "@lib/types";
import "./AskAi.css";

export interface AskAiProps {
  connectionId: number;
  onUse: (statement: string) => void;
}

export function AskAi({ connectionId, onUse }: AskAiProps) {
  const [open, setOpen] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [suggestion, setSuggestion] = useState<AiSuggestion | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [choice, setChoice] = useState<{ cli: string; model: string | null; installed: boolean } | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    setSuggestion(null);
    setError(null);
    api.aiSettings().then(setChoice).catch(() => setChoice(null));
    window.setTimeout(() => inputRef.current?.focus(), 0);
  }, [open]);

  const ask = async () => {
    if (!prompt.trim()) {
      return;
    }
    setBusy(true);
    setError(null);
    setSuggestion(null);
    try {
      setSuggestion(await api.aiQuery(connectionId, prompt));
    } catch (caught) {
      setError((caught as ApiError).message);
    } finally {
      setBusy(false);
    }
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape") {
      event.preventDefault();
      setOpen(false);
    }
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      void ask();
    }
  };

  const dialog = (
    <div className="askai-backdrop" onClick={() => setOpen(false)}>
      <div
        className="askai"
        role="dialog"
        aria-modal="true"
        aria-label="Ask AI for a query"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <header className="askai-header">
          <span className="askai-title">Ask AI for a query</span>
          {choice ? (
            <span className="askai-choice">
              <Badge tone="accent">{choice.cli}</Badge>
              {choice.model ? <span className="askai-model">{choice.model}</span> : null}
              <a href="/settings/ai">change</a>
            </span>
          ) : null}
        </header>

        <textarea
          ref={inputRef}
          className="askai-prompt"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="describe what you want to find, in plain language…"
          rows={3}
          aria-label="describe the query you want"
        />

        <div className="askai-actions">
          <span className="askai-hint">⌘↵ ask · esc close</span>
          <Button variant="primary" onClick={ask} disabled={busy || !prompt.trim()}>
            {busy ? "asking…" : "Ask"}
          </Button>
        </div>

        {error ? <p className="askai-error" role="alert">{error}</p> : null}

        {suggestion ? (
          <div className="askai-suggestion">
            <div className="askai-suggestion-head">
              <Badge tone={suggestion.readOnlyOk ? "ok" : "error"}>
                {suggestion.readOnlyOk ? "read only" : "rejected"}
              </Badge>
              <span className="askai-provenance">
                {suggestion.cli}
                {suggestion.model ? ` · ${suggestion.model}` : ""}
              </span>
            </div>
            <pre className="askai-statement">{suggestion.statement}</pre>
            {suggestion.denialReason ? (
              <p className="askai-denial">
                {suggestion.denialReason} — this suggestion cannot be run.
              </p>
            ) : null}
            <div className="askai-actions">
              <span className="askai-hint">nothing runs until you press Run</span>
              <Button
                variant="primary"
                disabled={!suggestion.readOnlyOk}
                onClick={() => {
                  onUse(suggestion.statement);
                  setOpen(false);
                }}
              >
                use this query
              </Button>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );

  return (
    <>
      <button className="askai-trigger" onClick={() => setOpen(true)}>
        ✦ ask ai
      </button>
      {open && typeof document !== "undefined" ? createPortal(dialog, document.body) : null}
    </>
  );
}
