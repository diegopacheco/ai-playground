import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { Stage } from "../engine/stage";
import { nextPage, prevPage } from "../story/navigation";
import { NARRATION_DELAY_MS, narrationParts } from "../story/narration";
import { PAGES } from "../story/pages";
import { canFullscreen, useFullscreen } from "./fullscreen";
import { canNarrate, hush, speak, unlock, voiceName } from "./narrator";

const SWIPE_PX = 50;

interface ShadowViewProps {
  page: number;
}

export function ShadowView({ page }: ShadowViewProps) {
  const host = useRef<HTMLDivElement>(null);
  const stage = useRef<Stage | null>(null);
  const [ready, setReady] = useState(false);
  const [narrating, setNarrating] = useState(false);
  const [peeking, setPeeking] = useState(false);
  const [fullscreen, toggleFullscreen] = useFullscreen();
  const navigate = useNavigate();
  const total = PAGES.length;
  const story = PAGES[page - 1];

  const go = useCallback(
    (target: number) => {
      void navigate({ to: "/page/$page", params: { page: String(target) } });
    },
    [navigate],
  );

  const toggleNarration = useCallback(() => {
    if (!narrating) unlock();
    setNarrating(!narrating);
  }, [narrating]);

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
    stage.current?.setPeek(peeking);
  }, [peeking]);

  useEffect(() => {
    if (!narrating || !ready || !story) return;
    const timer = window.setTimeout(() => speak(narrationParts(story)), NARRATION_DELAY_MS);
    return () => {
      window.clearTimeout(timer);
      hush();
    };
  }, [narrating, ready, story]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === " " && event.target instanceof HTMLButtonElement) return;
      if (event.key === "ArrowRight" || event.key === " ") go(nextPage(page, total));
      if (event.key === "ArrowLeft") go(prevPage(page, total));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, page, total]);

  useEffect(() => {
    let startX = 0;
    let startY = 0;
    const onStart = (event: TouchEvent) => {
      startX = event.touches[0]?.clientX ?? 0;
      startY = event.touches[0]?.clientY ?? 0;
    };
    const onEnd = (event: TouchEvent) => {
      const dx = (event.changedTouches[0]?.clientX ?? startX) - startX;
      const dy = (event.changedTouches[0]?.clientY ?? startY) - startY;
      if (Math.abs(dx) < SWIPE_PX || Math.abs(dx) < Math.abs(dy)) return;
      go(dx < 0 ? nextPage(page, total) : prevPage(page, total));
    };
    window.addEventListener("touchstart", onStart, { passive: true });
    window.addEventListener("touchend", onEnd, { passive: true });
    return () => {
      window.removeEventListener("touchstart", onStart);
      window.removeEventListener("touchend", onEnd);
    };
  }, [go, page, total]);

  return (
    <div className="shadow-view">
      <div className="stage" ref={host} />
      <div className="vignette" />
      {!ready && <div className="loading">Lighting the lamp...</div>}
      <header className="masthead">
        <span className="brand">The Shadow Theater presents</span>
        <span className="chapter">{story?.title}</span>
      </header>
      <div className="tools">
        {canNarrate && (
          <button className="tool" aria-pressed={narrating} title={voiceName()} onClick={toggleNarration}>
            {narrating ? "🔊 Narration on" : "🔈 Narration off"}
          </button>
        )}
        <button className="tool" aria-pressed={peeking} onClick={() => setPeeking(!peeking)}>
          {peeking ? "🎭 Back to the show" : "🪔 Behind the screen"}
        </button>
        {canFullscreen && (
          <button className="tool" aria-pressed={fullscreen} onClick={toggleFullscreen}>
            {fullscreen ? "⤡ Exit full screen" : "⤢ Full screen"}
          </button>
        )}
      </div>
      <button className="turn prev" aria-label="Previous scene" disabled={page === 1} onClick={() => go(prevPage(page, total))}>
        ‹
      </button>
      <button className="turn next" aria-label="Next scene" disabled={page === total} onClick={() => go(nextPage(page, total))}>
        ›
      </button>
      <footer className="caption" key={page}>
        {story?.lines.map((line) => <p key={line}>{line}</p>)}
        <div className="dots">
          {PAGES.map((p, i) => (
            <button key={p.scene} className={i + 1 === page ? "dot active" : "dot"} aria-label={`Scene ${i + 1}`} onClick={() => go(i + 1)} />
          ))}
        </div>
      </footer>
    </div>
  );
}
