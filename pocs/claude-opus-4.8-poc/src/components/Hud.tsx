import { useStore } from "@tanstack/react-store";
import { store } from "../game/store";
import { SHAPES, colorOf } from "../game/pieces";

export function Hud() {
  const score = useStore(store, (s) => s.score);
  const level = useStore(store, (s) => s.level);
  const cleared = useStore(store, (s) => s.cleared);
  const status = useStore(store, (s) => s.status);
  const nextShape = useStore(store, (s) => s.nextShape);
  const nextColor = colorOf(SHAPES[nextShape].color);

  return (
    <>
      <div className="panel stats">
        <h1>3D TETRIS</h1>
        <div className="row">
          <span>Score</span>
          <span>{score}</span>
        </div>
        <div className="row">
          <span>Level</span>
          <span>{level}</span>
        </div>
        <div className="row">
          <span>Layers</span>
          <span>{cleared}</span>
        </div>
        <div className="next">
          <span>Next</span>
          <span className="chip" style={{ background: nextColor }} />
        </div>
      </div>

      <div className="panel controls">
        <div>
          <b>Arrows</b> move on the floor
        </div>
        <div>
          <b>Q / E</b> rotate yaw · <b>A / D</b> rotate pitch · <b>W / S</b> rotate roll
        </div>
        <div>
          <b>Space</b> hard drop · <b>P</b> pause
        </div>
      </div>

      {status !== "playing" ? (
        <div className="overlay">
          {status === "idle" ? <h2>3D TETRIS</h2> : null}
          {status === "paused" ? <h2>PAUSED</h2> : null}
          {status === "over" ? <h2>GAME OVER</h2> : null}
          {status === "over" ? <p>Final score {score}</p> : null}
          <p className="hint">
            {status === "paused" ? "Press P to resume" : "Press Enter to play"}
          </p>
        </div>
      ) : null}
    </>
  );
}
