import { Button } from "../Button/Button";
import "./Pager.css";

export interface PagerProps {
  pageNumber: number;
  rowCount: number;
  elapsedMs?: number;
  hasMore: boolean;
  totalRows?: number | null;
  onFirst: () => void;
  onPrevious: () => void;
  onNext: () => void;
  onCount?: () => void;
}

export function Pager({
  pageNumber,
  rowCount,
  elapsedMs,
  hasMore,
  totalRows,
  onFirst,
  onPrevious,
  onNext,
  onCount
}: PagerProps) {
  const first = pageNumber <= 1;
  return (
    <div className="ds-pager">
      <span className="ds-pager-status">
        page {pageNumber} · {rowCount} rows
        {elapsedMs !== undefined ? ` · ${elapsedMs}ms` : ""}
        {totalRows !== null && totalRows !== undefined ? ` · ${totalRows} total` : ""}
      </span>
      <div className="ds-pager-controls">
        {onCount && totalRows === null ? (
          <Button variant="ghost" onClick={onCount} title="Runs a separate count query">
            count rows
          </Button>
        ) : null}
        <Button onClick={onFirst} disabled={first} aria-label="First page">
          ⇤ first
        </Button>
        <Button onClick={onPrevious} disabled={first} aria-label="Previous page">
          ‹ prev
        </Button>
        <Button onClick={onNext} disabled={!hasMore} aria-label="Next page">
          next ›
        </Button>
      </div>
    </div>
  );
}
