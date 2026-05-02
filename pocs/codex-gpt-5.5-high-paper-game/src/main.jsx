import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const moves = [
  { id: "paper", name: "Paper", mark: "P", beats: "rock" },
  { id: "cizer", name: "Cizer", mark: "C", beats: "paper" },
  { id: "rock", name: "Rock", mark: "R", beats: "cizer" }
];

const outcomes = {
  win: "You scored",
  lose: "House scored",
  draw: "Tie round"
};

function resultFor(player, house) {
  if (player.id === house.id) return "draw";
  return player.beats === house.id ? "win" : "lose";
}

function App() {
  const [player, setPlayer] = useState(null);
  const [house, setHouse] = useState(null);
  const [score, setScore] = useState({ player: 0, house: 0, ties: 0 });
  const [rounds, setRounds] = useState([]);

  const status = useMemo(() => {
    if (!player || !house) return "Pick a shape";
    return outcomes[resultFor(player, house)];
  }, [player, house]);

  function play(move) {
    const nextHouse = moves[Math.floor(Math.random() * moves.length)];
    const outcome = resultFor(move, nextHouse);
    setPlayer(move);
    setHouse(nextHouse);
    setScore((current) => ({
      player: current.player + (outcome === "win" ? 1 : 0),
      house: current.house + (outcome === "lose" ? 1 : 0),
      ties: current.ties + (outcome === "draw" ? 1 : 0)
    }));
    setRounds((current) => [
      { player: move.name, house: nextHouse.name, outcome },
      ...current.slice(0, 4)
    ]);
  }

  function reset() {
    setPlayer(null);
    setHouse(null);
    setScore({ player: 0, house: 0, ties: 0 });
    setRounds([]);
  }

  return (
    <main className="shell">
      <section className="stage">
        <div className="mast">
          <p className="kicker">Three shape duel</p>
          <h1>Paper Cizer Rock</h1>
        </div>

        <div className="scoreline" aria-label="Score">
          <div>
            <span>{score.player}</span>
            <small>You</small>
          </div>
          <div>
            <span>{score.ties}</span>
            <small>Ties</small>
          </div>
          <div>
            <span>{score.house}</span>
            <small>House</small>
          </div>
        </div>

        <div className="arena" aria-live="polite">
          <Choice title="You" move={player} side="left" />
          <div className="result">
            <span>{status}</span>
            <button type="button" onClick={reset}>Reset</button>
          </div>
          <Choice title="House" move={house} side="right" />
        </div>

        <div className="moves" aria-label="Moves">
          {moves.map((move) => (
            <button className={`move move-${move.id}`} type="button" key={move.id} onClick={() => play(move)}>
              <span>{move.mark}</span>
              {move.name}
            </button>
          ))}
        </div>
      </section>

      <aside className="ledger" aria-label="Round history">
        <h2>Rounds</h2>
        {rounds.length === 0 ? (
          <p className="empty">No rounds yet</p>
        ) : (
          <ol>
            {rounds.map((round, index) => (
              <li key={`${round.player}-${round.house}-${index}`}>
                <span>{round.player}</span>
                <b>{round.outcome}</b>
                <span>{round.house}</span>
              </li>
            ))}
          </ol>
        )}
      </aside>
    </main>
  );
}

function Choice({ title, move, side }) {
  return (
    <div className={`choice ${side} ${move ? `choice-${move.id}` : ""}`}>
      <small>{title}</small>
      <strong>{move?.mark || "?"}</strong>
      <span>{move?.name || "Waiting"}</span>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
