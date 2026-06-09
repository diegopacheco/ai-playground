"use client";

import { useState } from "react";
import {
  MAX_GUESSES,
  MAX_QUESTIONS,
  type Feedback,
  type GameRecord,
  type GameResult,
  type GuessResult,
  type Move,
  type Turn,
} from "@/lib/types";

const EXAMPLES = [
  "a Brazilian hotdog",
  "a rainy day in Japan",
  "a hot summer afternoon",
  "the smell of fresh coffee",
  "your first bike ride",
];

type Phase = "setup" | "thinking" | "awaiting" | "over";

export default function PlayPage() {
  const [phase, setPhase] = useState<Phase>("setup");
  const [secret, setSecret] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [pending, setPending] = useState<Move | null>(null);
  const [result, setResult] = useState<GameResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState("");

  const guessesUsed = turns.filter((t) => t.type === "guess").length;
  const questionsUsed = turns.filter((t) => t.type === "question").length;

  async function requestMove(history: Turn[]) {
    setPhase("thinking");
    setPending(null);
    setError(null);
    try {
      const res = await fetch("/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ turns: history }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "The AI got stuck.");
      setPending(data.move as Move);
      setPhase("awaiting");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
      setPhase("awaiting");
    }
  }

  function startGame() {
    const thing = secret.trim();
    if (!thing) return;
    const now = new Date().toISOString();
    setStartedAt(now);
    setTurns([]);
    setResult(null);
    setPhase("thinking");
    void requestMove([]);
  }

  async function finalize(finalTurns: Turn[], outcome: GameResult) {
    setResult(outcome);
    setPhase("over");
    const record: GameRecord = {
      id: crypto.randomUUID(),
      secret: secret.trim(),
      turns: finalTurns,
      result: outcome,
      guessesUsed: finalTurns.filter((t) => t.type === "guess").length,
      createdAt: startedAt || new Date().toISOString(),
      finishedAt: new Date().toISOString(),
    };
    try {
      await fetch("/api/history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(record),
      });
    } catch {
      /* history is best-effort */
    }
  }

  function answerQuestion(feedback: Feedback) {
    if (!pending) return;
    const done: Turn = { ...pending, response: feedback };
    const next = [...turns, done];
    setTurns(next);
    setPending(null);
    void requestMove(next);
  }

  function answerGuess(verdict: GuessResult) {
    if (!pending) return;
    const done: Turn = { ...pending, response: verdict };
    const next = [...turns, done];
    setTurns(next);
    setPending(null);
    if (verdict === "CORRECT") {
      void finalize(next, "win");
      return;
    }
    const used = next.filter((t) => t.type === "guess").length;
    if (used >= MAX_GUESSES) {
      void finalize(next, "loss");
      return;
    }
    void requestMove(next);
  }

  function reset() {
    setPhase("setup");
    setSecret("");
    setTurns([]);
    setPending(null);
    setResult(null);
    setError(null);
  }

  if (phase === "setup") {
    return (
      <section className="card hero">
        <h1>Think of something. I&apos;ll read your mind.</h1>
        <p>
          Picture a thing, a place or a feeling. I get{" "}
          {`${MAX_QUESTIONS} questions`}, then I&apos;m forced to guess — and only{" "}
          {`${MAX_GUESSES} guesses`} to nail it. All you ever say back is 🔥 HOT or 🧊
          COLD.
        </p>
        <div className="start-row">
          <input
            className="text-input"
            placeholder="Type your secret thing…"
            value={secret}
            onChange={(e) => setSecret(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && startGame()}
            autoFocus
          />
          <button className="btn btn-accent" onClick={startGame} disabled={!secret.trim()}>
            Start
          </button>
        </div>
        <p className="hint">
          Your secret stays on this screen — the AI never sees it, it only feels the
          heat. Need inspiration?
        </p>
        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="chip" onClick={() => setSecret(ex)}>
              {ex}
            </button>
          ))}
        </div>
      </section>
    );
  }

  return (
    <section className="card">
      <div className="status-row">
        <span className="secret-pill">🤫 {secret}</span>
        <div className="counters">
          <span
            className={`qcount ${questionsUsed >= MAX_QUESTIONS ? "maxed" : ""}`}
          >
            ❓ {questionsUsed}/{MAX_QUESTIONS}
          </span>
          <div className="guess-dots">
            <span className="guess-label">Guesses</span>
            {Array.from({ length: MAX_GUESSES }).map((_, i) => (
              <span key={i} className={`dot ${i < guessesUsed ? "used" : ""}`} />
            ))}
          </div>
        </div>
      </div>

      <div className="transcript">
        {turns.map((turn, i) => (
          <ReadOnlyTurn key={i} turn={turn} />
        ))}

        {pending && phase === "awaiting" && (
          <div className={`bubble bubble-ai ${pending.type === "guess" ? "is-guess" : ""}`}>
            <div className="bubble-meta">
              {pending.type === "guess" ? "🎯 My guess" : "🤖 I ask"}
            </div>
            {pending.text}
          </div>
        )}

        {phase === "thinking" && (
          <div className="thinking">
            <span />
            <span />
            <span />
          </div>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      {phase === "awaiting" && pending && pending.type === "question" && (
        <div className="action-bar">
          <button className="big-btn hot" onClick={() => answerQuestion("HOT")}>
            🔥 HOT
          </button>
          <button className="big-btn cold" onClick={() => answerQuestion("COLD")}>
            🧊 COLD
          </button>
        </div>
      )}

      {phase === "awaiting" && pending && pending.type === "guess" && (
        <div className="action-bar">
          <button className="big-btn correct" onClick={() => answerGuess("CORRECT")}>
            ✅ Nailed it
          </button>
          <button className="big-btn wrong" onClick={() => answerGuess("WRONG")}>
            ❌ Nope
          </button>
        </div>
      )}

      {phase === "over" && result === "win" && (
        <div className="banner win">
          <h2>🎉 Read your mind!</h2>
          <p>
            The AI guessed “{secret}” in {guessesUsed} of {MAX_GUESSES} guesses.
          </p>
          <button className="btn btn-accent" onClick={reset}>
            Play again
          </button>
        </div>
      )}

      {phase === "over" && result === "loss" && (
        <div className="banner loss">
          <h2>🧊 You stumped the AI!</h2>
          <p>
            {MAX_GUESSES} guesses gone and “{secret}” stayed a secret. You win this round.
          </p>
          <button className="btn btn-accent" onClick={reset}>
            Play again
          </button>
        </div>
      )}
    </section>
  );
}

function ReadOnlyTurn({ turn }: { turn: Turn }) {
  const reactionClass =
    turn.response === "HOT"
      ? "hot"
      : turn.response === "COLD"
        ? "cold"
        : turn.response === "CORRECT"
          ? "correct"
          : "wrong";
  const reactionText =
    turn.response === "HOT"
      ? "🔥 HOT"
      : turn.response === "COLD"
        ? "🧊 COLD"
        : turn.response === "CORRECT"
          ? "✅ Nailed it"
          : "❌ Nope";
  return (
    <>
      <div className={`bubble bubble-ai ${turn.type === "guess" ? "is-guess" : ""}`}>
        <div className="bubble-meta">
          {turn.type === "guess" ? "🎯 My guess" : "🤖 I ask"}
        </div>
        {turn.text}
      </div>
      <div className={`bubble bubble-you ${reactionClass}`}>{reactionText}</div>
    </>
  );
}
