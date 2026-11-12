import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { Stage } from "../engine/stage";
import { nextPage, prevPage } from "../story/navigation";
import { PAGES } from "../story/pages";

interface BookViewProps {
  page: number;
}

export function BookView({ page }: BookViewProps) {
  const host = useRef<HTMLDivElement>(null);
  const stage = useRef<Stage | null>(null);
  const [ready, setReady] = useState(false);
  const navigate = useNavigate();
  const total = PAGES.length;
  const story = PAGES[page - 1];

  const go = useCallback(
    (target: number) => {
      void navigate({ to: "/page/$page", params: { page: String(target) } });
    },
    [navigate],
  );

  useEffect(() => {
    if (!host.current) return;
    const instance = new Stage(host.current, () => setReady(true));
    stage.current = instance;
    void instance.start(page);
    return () => {
      instance.dispose();
      stage.current = null;
    };
  }, []);

  useEffect(() => {
    stage.current?.goTo(page);
  }, [page]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowRight" || event.key === " ") go(nextPage(page, total));
      if (event.key === "ArrowLeft") go(prevPage(page, total));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, page, total]);

  return (
    <div className="book-view">
      <div className="stage" ref={host} />
      <div className="vignette" />
      {!ready && <div className="loading">Opening the book...</div>}
      <header className="masthead">
        <span className="brand">The Three Little Pigs</span>
        <span className="chapter">{story?.title}</span>
      </header>
      <button className="turn prev" aria-label="Previous page" disabled={page === 1} onClick={() => go(prevPage(page, total))}>
        ‹
      </button>
      <button className="turn next" aria-label="Next page" disabled={page === total} onClick={() => go(nextPage(page, total))}>
        ›
      </button>
      <footer className="caption" key={page}>
        {story?.lines.map((line) => <p key={line}>{line}</p>)}
        <div className="dots">
          {PAGES.map((p, i) => (
            <button key={p.scene} className={i + 1 === page ? "dot active" : "dot"} aria-label={`Page ${i + 1}`} onClick={() => go(i + 1)} />
          ))}
        </div>
      </footer>
    </div>
  );
}
