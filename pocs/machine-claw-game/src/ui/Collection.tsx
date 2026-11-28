import { useEffect, useRef } from "react";
import { type Collection, collectionEntries, totalWon } from "../game/collection";
import { PLUSH_LOOKS } from "../game/plush";
import { createShowcase } from "../render/showcase";

type CollectionProps = {
  readonly collection: Collection;
  readonly onClose: () => void;
};

export function CollectionView({ collection, onClose }: CollectionProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const entries = collectionEntries(collection);
  const total = totalWon(collection);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || entries.length === 0) return;
    const showcase = createShowcase(canvas, entries);
    return () => showcase.dispose();
  }, [collection]);

  useEffect(() => {
    function key(event: KeyboardEvent): void {
      if (event.code === "Escape") onClose();
    }
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);

  return (
    <section className="collection" role="dialog" aria-modal="true" aria-label="My Stuffies">
      <header className="collection-top">
        <div>
          <h2>My Stuffies</h2>
          <p>
            {total} won - {entries.length} of {PLUSH_LOOKS.length} kinds
          </p>
        </div>
        <button type="button" className="collection-close" onClick={onClose} aria-label="Close">
          ×
        </button>
      </header>
      {entries.length > 0 ? (
        <>
          <canvas ref={canvasRef} className="collection-stage" />
          <p className="collection-hint">Drag to spin your stuffies</p>
        </>
      ) : (
        <div className="collection-empty">
          <span>🧸</span>
          <p>No stuffies yet. Win one with the claw and it will live here.</p>
        </div>
      )}
    </section>
  );
}
