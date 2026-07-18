import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api } from "@lib/api";
import "./RecentQueries.css";

export interface RecentQueriesProps {
  connectionId: number;
  onPick: (statement: string) => void;
}

export function RecentQueries({ connectionId, onPick }: RecentQueriesProps) {
  const [statements, setStatements] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const [anchor, setAnchor] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    api
      .history(connectionId)
      .then(setStatements)
      .catch(() => setStatements([]));
  }, [connectionId, open]);

  useLayoutEffect(() => {
    if (!open || !triggerRef.current) {
      return;
    }
    const rect = triggerRef.current.getBoundingClientRect();
    setAnchor({ top: rect.bottom + 6, left: rect.left });
  }, [open]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const close = (event: MouseEvent) => {
      const target = event.target as Node;
      const insideTrigger = triggerRef.current?.contains(target);
      const insideList = listRef.current?.contains(target);
      if (!insideTrigger && !insideList) {
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

  const list = (
    <ul
      ref={listRef}
      className="recent-queries-list"
      role="listbox"
      aria-label="recent queries"
      style={{ top: anchor.top, left: anchor.left }}
    >
      {statements.length === 0 ? (
        <li className="recent-queries-empty">nothing yet</li>
      ) : (
        statements.map((statement) => (
          <li key={statement}>
            <button
              role="option"
              aria-selected={false}
              onClick={() => {
                onPick(statement);
                setOpen(false);
              }}
              title={statement}
            >
              {statement.length > 70 ? `${statement.slice(0, 70)}…` : statement}
            </button>
          </li>
        ))
      )}
    </ul>
  );

  return (
    <div className="recent-queries">
      <button
        ref={triggerRef}
        className="recent-queries-toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        recent ▾
      </button>
      {open && typeof document !== "undefined" ? createPortal(list, document.body) : null}
    </div>
  );
}
