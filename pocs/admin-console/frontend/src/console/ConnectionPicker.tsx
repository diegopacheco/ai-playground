import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { createPortal } from "react-dom";
import { Badge } from "@design/Badge/Badge";
import { EngineLogo } from "@design/EngineLogo/EngineLogo";
import { isGridKey, measureColumns, nextIndex } from "@lib/gridKeys";
import type { Connection } from "@lib/types";
import "./ConnectionPicker.css";

export interface ConnectionPickerProps {
  connections: Connection[];
  selected: Connection | null;
  onSelect: (connection: Connection) => void;
  autoOpen?: boolean;
}

function matches(connection: Connection, term: string) {
  if (!term) {
    return true;
  }
  const haystack = [connection.name, connection.kind, connection.host, connection.database ?? "", connection.keyspace ?? ""]
    .join(" ")
    .toLowerCase();
  return haystack.includes(term.toLowerCase());
}

export function ConnectionPicker({ connections, selected, onSelect, autoOpen = false }: ConnectionPickerProps) {
  const [open, setOpen] = useState(autoOpen);
  const [term, setTerm] = useState("");
  const [active, setActive] = useState(0);
  const searchRef = useRef<HTMLInputElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(
    () => connections.filter((connection) => matches(connection, term)),
    [connections, term]
  );

  useEffect(() => {
    setActive(0);
  }, [term]);

  useEffect(() => {
    if (open) {
      setTerm("");
      setActive(0);
      window.setTimeout(() => searchRef.current?.focus(), 0);
    }
  }, [open]);

  const choose = (connection: Connection | undefined) => {
    if (!connection) {
      return;
    }
    onSelect(connection);
    setOpen(false);
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape") {
      event.preventDefault();
      setOpen(false);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      choose(filtered[active]);
      return;
    }
    if (event.key === "Tab") {
      event.preventDefault();
      const step = event.shiftKey ? -1 : 1;
      setActive((current) =>
        filtered.length === 0 ? 0 : (current + step + filtered.length) % filtered.length
      );
      return;
    }
    if (isGridKey(event.key)) {
      event.preventDefault();
      const columns = measureColumns(gridRef.current, ".picker-card");
      setActive((current) => nextIndex(event.key as never, current, filtered.length, columns));
    }
  };

  return (
    <>
      <button className="picker-trigger" onClick={() => setOpen(true)} aria-haspopup="dialog">
        {selected ? (
          <>
            <EngineLogo kind={selected.kind} size={18} />
            <span className="picker-trigger-name">{selected.name}</span>
            <Badge tone="accent">{selected.kind}</Badge>
          </>
        ) : (
          <span className="picker-trigger-name">choose a connection</span>
        )}
        <span className="picker-trigger-caret">▾</span>
      </button>

      {open && typeof document !== "undefined" ? createPortal(
        <div className="picker-backdrop" onClick={() => setOpen(false)}>
          <div
            className="picker-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Choose a connection"
            onClick={(event) => event.stopPropagation()}
            onKeyDown={onKeyDown}
          >
            <header className="picker-header">
              <input
                ref={searchRef}
                className="picker-search"
                value={term}
                onChange={(event) => setTerm(event.target.value)}
                placeholder="search connections, engines, hosts…"
                aria-label="search connections"
              />
              <span className="picker-hint">↑↓←→ move · ↵ open · esc close</span>
            </header>

            <div ref={gridRef} className="picker-grid" role="listbox" aria-label="connections">
              {filtered.length === 0 ? (
                <p className="picker-empty">no connection matches “{term}”</p>
              ) : (
                filtered.map((connection, index) => (
                  <button
                    key={connection.id}
                    role="option"
                    aria-selected={index === active}
                    className={index === active ? "picker-card picker-card-active" : "picker-card"}
                    onClick={() => choose(connection)}
                    onMouseEnter={() => setActive(index)}
                  >
                    <EngineLogo kind={connection.kind} size={30} />
                    <span className="picker-card-name">{connection.name}</span>
                    <Badge tone={connection.id === selected?.id ? "accent" : "neutral"}>{connection.kind}</Badge>
                    <span className="picker-card-target">
                      {connection.host}:{connection.port}
                      {connection.database ? ` / ${connection.database}` : ""}
                      {connection.keyspace ? ` / ${connection.keyspace}` : ""}
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>,
        document.body
      ) : null}
    </>
  );
}
