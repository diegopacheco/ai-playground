import { useEffect, useState, type KeyboardEvent } from "react";
import { createPortal } from "react-dom";
import { Button } from "../Button/Button";
import type { GridRow } from "../DataGrid/DataGrid";
import "./RowDetail.css";

export interface RowDetailProps {
  columns: string[];
  rows: GridRow[];
  index: number;
  onClose: () => void;
  onNavigate: (index: number) => void;
}

function render(value: unknown) {
  if (value === null || value === undefined) {
    return { text: "null", isNull: true };
  }
  const text = String(value);
  const trimmed = text.trim();
  if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
    try {
      return { text: JSON.stringify(JSON.parse(trimmed), null, 2), isNull: false };
    } catch {
      return { text, isNull: false };
    }
  }
  return { text, isNull: false };
}

export function RowDetail({ columns, rows, index, onClose, onNavigate }: RowDetailProps) {
  const [copied, setCopied] = useState<string | null>(null);
  const row = rows[index];

  useEffect(() => {
    setCopied(null);
  }, [index]);

  if (!row) {
    return null;
  }

  const copy = async (label: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
    } catch {
      setCopied(null);
    }
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
    }
    if (event.key === "ArrowDown" && index < rows.length - 1) {
      event.preventDefault();
      onNavigate(index + 1);
    }
    if (event.key === "ArrowUp" && index > 0) {
      event.preventDefault();
      onNavigate(index - 1);
    }
  };

  const modal = (
    <div className="row-detail-backdrop" onClick={onClose}>
      <div
        className="row-detail"
        role="dialog"
        aria-modal="true"
        aria-label={`Row ${index + 1} details`}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={onKeyDown}
        tabIndex={-1}
        ref={(node) => node?.focus()}
      >
        <header className="row-detail-header">
          <span className="row-detail-title">
            row {index + 1} <span className="row-detail-of">of {rows.length} on this page</span>
          </span>
          <div className="row-detail-nav">
            <Button variant="ghost" onClick={() => onNavigate(index - 1)} disabled={index === 0} aria-label="Previous row">
              ↑
            </Button>
            <Button
              variant="ghost"
              onClick={() => onNavigate(index + 1)}
              disabled={index >= rows.length - 1}
              aria-label="Next row"
            >
              ↓
            </Button>
            <Button
              variant="ghost"
              onClick={() => copy("row", JSON.stringify(row, null, 2))}
              aria-label="Copy row as JSON"
            >
              {copied === "row" ? "copied" : "copy json"}
            </Button>
            <Button variant="ghost" onClick={onClose} aria-label="Close row details">
              ✕
            </Button>
          </div>
        </header>

        <dl className="row-detail-fields">
          {columns.map((column) => {
            const { text, isNull } = render(row[column]);
            const multiline = text.includes("\n") || text.length > 90;
            return (
              <div className="row-detail-field" key={column}>
                <dt>{column}</dt>
                <dd>
                  <div className={multiline ? "row-detail-value row-detail-value-block" : "row-detail-value"}>
                    {isNull ? <span className="row-detail-null">null</span> : text}
                  </div>
                  <button
                    className="row-detail-copy"
                    onClick={() => copy(column, text)}
                    aria-label={`Copy ${column}`}
                  >
                    {copied === column ? "copied" : "copy"}
                  </button>
                </dd>
              </div>
            );
          })}
        </dl>

        <footer className="row-detail-footer">↑↓ move between rows · esc close</footer>
      </div>
    </div>
  );

  return typeof document === "undefined" ? modal : createPortal(modal, document.body);
}
