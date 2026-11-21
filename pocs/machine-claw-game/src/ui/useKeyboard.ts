import { useEffect } from "react";
import type { Engine } from "../game/engine";

const AXIS: Readonly<Record<string, readonly [x: number, z: number]>> = {
  ArrowLeft: [-1, 0],
  KeyA: [-1, 0],
  ArrowRight: [1, 0],
  KeyD: [1, 0],
  ArrowUp: [0, -1],
  KeyW: [0, -1],
  ArrowDown: [0, 1],
  KeyS: [0, 1],
};

export function useKeyboard(engine: Engine | null): void {
  useEffect(() => {
    if (!engine) return;
    const held = new Set<string>();

    function steer(): void {
      let x = 0;
      let z = 0;
      for (const code of held) {
        const axis = AXIS[code];
        if (!axis) continue;
        x += axis[0];
        z += axis[1];
      }
      engine?.steer(Math.sign(x), Math.sign(z));
    }

    function down(event: KeyboardEvent): void {
      if (event.repeat) return;
      if (AXIS[event.code]) {
        held.add(event.code);
        steer();
      } else if (event.code === "Space") engine?.drop();
      else if (event.code === "KeyC") engine?.insertCoin();
      else if (event.code === "KeyV") engine?.toggleView();
      else return;
      event.preventDefault();
    }

    function up(event: KeyboardEvent): void {
      if (held.delete(event.code)) steer();
    }

    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [engine]);
}
