import { useEffect } from "react";
import { Scene } from "./components/Scene";
import { Hud } from "./components/Hud";
import { store, start, pause, move, rotate, drop, tick, intervalFor } from "./game/store";

const HANDLED = new Set([
  "ArrowLeft",
  "ArrowRight",
  "ArrowUp",
  "ArrowDown",
  " ",
]);

export function App() {
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const loop = (now: number) => {
      const s = store.state;
      if (s.status === "playing") {
        acc += now - last;
        const iv = intervalFor(s.level);
        while (acc >= iv) {
          acc -= iv;
          tick();
        }
      } else {
        acc = 0;
      }
      last = now;
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (HANDLED.has(e.key)) e.preventDefault();
      switch (e.key) {
        case "Enter":
          if (store.state.status !== "playing") start();
          break;
        case "p":
        case "P":
          pause();
          break;
        case "ArrowLeft":
          move(-1, 0, 0);
          break;
        case "ArrowRight":
          move(1, 0, 0);
          break;
        case "ArrowUp":
          move(0, 0, -1);
          break;
        case "ArrowDown":
          move(0, 0, 1);
          break;
        case "q":
        case "Q":
          rotate("y", 1);
          break;
        case "e":
        case "E":
          rotate("y", -1);
          break;
        case "a":
        case "A":
          rotate("x", 1);
          break;
        case "d":
        case "D":
          rotate("x", -1);
          break;
        case "w":
        case "W":
          rotate("z", 1);
          break;
        case "s":
        case "S":
          rotate("z", -1);
          break;
        case " ":
          drop();
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="app">
      <Scene />
      <Hud />
    </div>
  );
}
