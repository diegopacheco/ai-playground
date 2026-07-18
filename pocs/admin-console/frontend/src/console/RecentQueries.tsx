import { useEffect, useState } from "react";
import { api } from "@lib/api";
import "./RecentQueries.css";

export interface RecentQueriesProps {
  connectionId: number;
  onPick: (statement: string) => void;
}

export function RecentQueries({ connectionId, onPick }: RecentQueriesProps) {
  const [statements, setStatements] = useState<string[]>([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) {
      return;
    }
    api
      .history(connectionId)
      .then(setStatements)
      .catch(() => setStatements([]));
  }, [connectionId, open]);

  return (
    <div className="recent-queries">
      <button className="recent-queries-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        recent ▾
      </button>
      {open ? (
        <ul className="recent-queries-list" role="listbox">
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
      ) : null}
    </div>
  );
}
