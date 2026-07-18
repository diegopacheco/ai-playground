import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Badge } from "@design/Badge/Badge";
import { Button } from "@design/Button/Button";
import { api } from "@lib/api";
import type { ApiError, SavedQuery } from "@lib/types";
import "./SavedQueries.css";

export interface SavedQueriesProps {
  projectId: number;
  connectionId: number;
  connectionKind: string;
  statement: string;
  onUse: (statement: string) => void;
}

export function SavedQueries({ projectId, connectionId, connectionKind, statement, onUse }: SavedQueriesProps) {
  const [open, setOpen] = useState(false);
  const [queries, setQueries] = useState<SavedQuery[]>([]);
  const [name, setName] = useState("");
  const [pinned, setPinned] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [anchor, setAnchor] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  const load = useCallback(() => {
    api
      .savedQueries(projectId)
      .then(setQueries)
      .catch((caught: ApiError) => setError(caught.message));
  }, [projectId]);

  useEffect(() => {
    if (!open) {
      return;
    }
    setError(null);
    load();
    const rect = triggerRef.current?.getBoundingClientRect();
    if (rect) {
      setAnchor({ top: rect.bottom + 6, left: Math.max(rect.left - 180, 12) });
    }
  }, [open, load]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const close = (event: MouseEvent) => {
      const target = event.target as Node;
      if (!triggerRef.current?.contains(target) && !panelRef.current?.contains(target)) {
        setOpen(false);
      }
    };
    const onEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", onEscape);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", onEscape);
    };
  }, [open]);

  const save = async () => {
    if (!name.trim() || !statement.trim()) {
      return;
    }
    setError(null);
    try {
      await api.saveQuery(projectId, {
        name: name.trim(),
        statement,
        connectionId: pinned ? connectionId : null,
        description: null
      });
      setName("");
      load();
    } catch (caught) {
      setError((caught as ApiError).message);
    }
  };

  const remove = async (id: number) => {
    try {
      await api.deleteSavedQuery(projectId, id);
      load();
    } catch (caught) {
      setError((caught as ApiError).message);
    }
  };

  const usable = (query: SavedQuery) =>
    query.connectionId === null || query.connectionId === connectionId || query.kind === connectionKind;

  const panel = (
    <div ref={panelRef} className="saved-panel" style={{ top: anchor.top, left: anchor.left }}>
      <header className="saved-panel-head">
        <span>saved in this project</span>
        <Badge>{queries.length}</Badge>
      </header>

      {error ? <p className="saved-error" role="alert">{error}</p> : null}

      <ul className="saved-list">
        {queries.length === 0 ? (
          <li className="saved-empty">nothing saved yet</li>
        ) : (
          queries.map((query) => (
            <li key={query.id} className={usable(query) ? "saved-item" : "saved-item saved-item-foreign"}>
              <button
                className="saved-item-main"
                onClick={() => {
                  onUse(query.statement);
                  setOpen(false);
                }}
                title={query.statement}
              >
                <span className="saved-item-name">{query.name}</span>
                <span className="saved-item-statement">{query.statement.replace(/\s+/g, " ").slice(0, 64)}</span>
              </button>
              <Badge tone={query.connectionId === null ? "neutral" : "accent"}>
                {query.connectionId === null ? "any" : query.kind}
              </Badge>
              <button className="saved-item-remove" onClick={() => remove(query.id)} aria-label={`Delete ${query.name}`}>
                ✕
              </button>
            </li>
          ))
        )}
      </ul>

      <footer className="saved-save">
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="save current statement as…"
          aria-label="saved query name"
        />
        <label className="saved-pin">
          <input type="checkbox" checked={pinned} onChange={(event) => setPinned(event.target.checked)} />
          pin to this connection
        </label>
        <Button variant="primary" onClick={save} disabled={!name.trim() || !statement.trim()}>
          save
        </Button>
      </footer>
    </div>
  );

  return (
    <div className="saved-queries">
      <button
        ref={triggerRef}
        className="saved-trigger"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        ★ saved
      </button>
      {open && typeof document !== "undefined" ? createPortal(panel, document.body) : null}
    </div>
  );
}
