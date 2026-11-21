import { useSyncExternalStore } from "react";
import type { Banner, Engine } from "../game/engine";
import { PLUSH_LOOKS } from "../game/plush";
import { Joystick } from "./Joystick";

type HudProps = { readonly engine: Engine };

const BANNERS: Readonly<Record<Banner, string>> = {
  insert: "INSERT COIN",
  aim: "MOVE THE CLAW THEN DROP",
  watch: "GOOD LUCK",
  won: "YOU WON A STUFFY",
  miss: "SO CLOSE - TRY AGAIN",
};

export function Hud({ engine }: HudProps) {
  const hud = useSyncExternalStore(engine.subscribe, engine.snapshot);
  const aiming = hud.phase === "aiming";

  return (
    <div className="hud">
      <header className="stats">
        <div className="stat">
          <span>CREDITS</span>
          <strong>{hud.credits}</strong>
        </div>
        <div className={aiming && hud.seconds <= 5 ? "stat stat-hot" : "stat"}>
          <span>TIME</span>
          <strong>{aiming ? hud.seconds : "--"}</strong>
        </div>
        <div className="stat">
          <span>PRIZES</span>
          <strong>{hud.prizes}</strong>
        </div>
        <button type="button" className="view" onClick={engine.toggleView} aria-label="Switch camera">
          VIEW
        </button>
      </header>
      <div className={`banner banner-${hud.banner}`} role="status">
        {BANNERS[hud.banner]}
      </div>
      <ul className="shelf" aria-label="Prizes won">
        {hud.shelf.map((look, index) => (
          <li key={index}>{PLUSH_LOOKS[look]?.emoji}</li>
        ))}
      </ul>
      <footer className="controls">
        <Joystick disabled={!aiming} onMove={engine.steer} />
        <div className="buttons">
          <button type="button" className="coin" onClick={engine.insertCoin}>
            COIN
          </button>
          <button type="button" className="drop" disabled={!aiming} onClick={engine.drop}>
            DROP
          </button>
        </div>
      </footer>
      <p className="keys">Arrows / WASD move - Space drop - C coin - V view</p>
    </div>
  );
}
