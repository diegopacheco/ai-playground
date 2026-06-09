import Link from "next/link";
import { listGames } from "@/lib/store";
import { MAX_GUESSES, type Turn } from "@/lib/types";

export const dynamic = "force-dynamic";

function reactionText(turn: Turn): string {
  switch (turn.response) {
    case "HOT":
      return "🔥 HOT";
    case "COLD":
      return "🧊 COLD";
    case "CORRECT":
      return "✅ Nailed it";
    case "WRONG":
      return "❌ Nope";
    default:
      return "";
  }
}

export default async function HistoryPage() {
  const games = await listGames();
  const total = games.length;
  const aiWins = games.filter((g) => g.result === "win").length;
  const humanWins = total - aiWins;
  const winRate = total ? Math.round((aiWins / total) * 100) : 0;

  return (
    <section>
      <div className="history-head">
        <h1>📜 Game History</h1>
        <p>Every mind the AI tried to read. Expand a row to replay the duel.</p>
      </div>

      <div className="stat-row">
        <div className="stat">
          <div className="num">{total}</div>
          <div className="lbl">Games played</div>
        </div>
        <div className="stat">
          <div className="num">{aiWins}</div>
          <div className="lbl">AI wins 🤖</div>
        </div>
        <div className="stat">
          <div className="num">{humanWins}</div>
          <div className="lbl">Your wins 🧠</div>
        </div>
        <div className="stat">
          <div className="num">{winRate}%</div>
          <div className="lbl">AI win rate</div>
        </div>
      </div>

      {total === 0 ? (
        <div className="empty">
          <div className="big">🪄</div>
          <p>No games yet. </p>
          <Link href="/" className="btn btn-accent">
            Play the first round
          </Link>
        </div>
      ) : (
        <div className="game-list">
          {games.map((game) => (
            <details key={game.id} className="game-row">
              <summary>
                <div>
                  <div className="game-secret">🤫 {game.secret}</div>
                  <div className="game-sub">
                    {game.guessesUsed} / {MAX_GUESSES} guesses ·{" "}
                    {game.turns.length} moves ·{" "}
                    {new Date(game.createdAt).toLocaleString()}
                  </div>
                </div>
                <span className={`tag ${game.result === "win" ? "win" : "loss"}`}>
                  {game.result === "win" ? "AI won 🤖" : "You won 🧠"}
                </span>
              </summary>
              <div className="game-detail">
                {game.turns.map((turn, i) => (
                  <div key={i} className="move">
                    <span className={`kind ${turn.type}`}>{turn.type}</span>
                    <span>{turn.text}</span>
                    <span className="react">{reactionText(turn)}</span>
                  </div>
                ))}
              </div>
            </details>
          ))}
        </div>
      )}
    </section>
  );
}
