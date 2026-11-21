import { useEffect, useRef, useState } from "react";
import { type Engine, createEngine } from "../game/engine";
import { logger } from "../logger";
import { Hud } from "./Hud";
import { useKeyboard } from "./useKeyboard";

export function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [engine, setEngine] = useState<Engine | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    let created: Engine | null = null;
    createEngine(canvas)
      .then((made) => {
        if (cancelled) {
          made.dispose();
          return;
        }
        created = made;
        setEngine(made);
      })
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        logger.error("engine failed", { message });
        setFailure(message);
      });
    return () => {
      cancelled = true;
      created?.dispose();
      setEngine(null);
    };
  }, []);

  useKeyboard(engine);

  return (
    <>
      <canvas ref={canvasRef} className="stage" />
      {engine ? <Hud engine={engine} /> : <div className="loading">{failure ?? "Warming up the claw..."}</div>}
    </>
  );
}
