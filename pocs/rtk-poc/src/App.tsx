import { useCallback, useEffect, useMemo, useState } from "react";
import { type ApiCard, type ApiScore, getScores, newGame, submitScore } from "./api.ts";

type Card = ApiCard & { matched: boolean };

function toLocalDeck(deck: ApiCard[]): Card[] {
  return deck.map((c) => ({ ...c, matched: false }));
}

export function App() {
  const [deck, setDeck] = useState<Card[]>([]);
  const [revealed, setRevealed] = useState<number[]>([]);
  const [moves, setMoves] = useState(0);
  const [locked, setLocked] = useState(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [scores, setScores] = useState<ApiScore[]>([]);
  const [name, setName] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = useCallback(async () => {
    setError(null);
    try {
      const fresh = await newGame();
      setDeck(toLocalDeck(fresh));
      setRevealed([]);
      setMoves(0);
      setLocked(false);
      setStartedAt(null);
      setElapsed(0);
      setSubmitted(false);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  const refreshScores = useCallback(async () => {
    try {
      setScores(await getScores());
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    reset();
    refreshScores();
  }, [reset, refreshScores]);

  const won = useMemo(
    () => deck.length > 0 && deck.every((c) => c.matched),
    [deck],
  );

  useEffect(() => {
    if (won || startedAt === null) return;
    const t = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    }, 250);
    return () => clearInterval(t);
  }, [won, startedAt]);

  useEffect(() => {
    if (revealed.length !== 2) return;
    setLocked(true);
    const [a, b] = revealed;
    const cardA = deck.find((c) => c.id === a)!;
    const cardB = deck.find((c) => c.id === b)!;
    if (cardA.symbol === cardB.symbol) {
      setDeck((d) =>
        d.map((c) => (c.id === a || c.id === b ? { ...c, matched: true } : c))
      );
      setRevealed([]);
      setLocked(false);
    } else {
      const timer = setTimeout(() => {
        setRevealed([]);
        setLocked(false);
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [revealed, deck]);

  function handleClick(card: Card) {
    if (locked || card.matched || revealed.includes(card.id)) return;
    if (startedAt === null) setStartedAt(Date.now());
    if (revealed.length === 0) {
      setRevealed([card.id]);
      setMoves((m) => m + 1);
    } else if (revealed.length === 1) {
      setRevealed([revealed[0], card.id]);
    }
  }

  async function handleSubmit() {
    if (!name.trim() || submitted || !won) return;
    try {
      await submitScore({ name: name.trim(), moves, seconds: Math.max(1, elapsed) });
      setSubmitted(true);
      await refreshScores();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const matchedCount = deck.filter((c) => c.matched).length / 2;

  return (
    <div className="game">
      <header className="header">
        <h1>Memory Game</h1>
        <div className="stats">
          <span>Moves: {moves}</span>
          <span>Time: {elapsed}s</span>
          <span>Pairs: {matchedCount} / {deck.length / 2 || 0}</span>
        </div>
      </header>
      <div className="controls">
        <button onClick={reset}>New Game</button>
      </div>
      {error && <div className="error">{error}</div>}
      {won && !submitted && (
        <div className="win">
          <p>You won in {moves} moves and {elapsed}s!</p>
          <div className="submit-row">
            <input
              type="text"
              placeholder="Your name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={24}
            />
            <button onClick={handleSubmit} disabled={!name.trim()}>
              Save Score
            </button>
          </div>
        </div>
      )}
      {won && submitted && <div className="win">Score saved.</div>}
      <div className="board">
        {deck.map((card) => {
          const isFlipped = card.matched || revealed.includes(card.id);
          const classes = [
            "card",
            isFlipped ? "flipped" : "",
            card.matched ? "matched" : "",
          ]
            .filter(Boolean)
            .join(" ");
          return (
            <div
              key={card.id}
              className={classes}
              onClick={() => handleClick(card)}
            >
              <div className="card-inner">
                <div className="card-face card-front">?</div>
                <div className="card-face card-back">{card.symbol}</div>
              </div>
            </div>
          );
        })}
      </div>
      <section className="leaderboard">
        <h2>Leaderboard</h2>
        {scores.length === 0 && <p className="empty">No scores yet.</p>}
        {scores.length > 0 && (
          <ol>
            {scores.map((s, i) => (
              <li key={i}>
                <span>{s.name}</span>
                <span>{s.moves} moves · {s.seconds}s</span>
              </li>
            ))}
          </ol>
        )}
      </section>
    </div>
  );
}
