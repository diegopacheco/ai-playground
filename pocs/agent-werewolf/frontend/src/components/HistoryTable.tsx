"use client";
import { useState, useEffect } from "react";
import { Game, AGENT_COLORS } from "@/types";
import { getGames } from "@/lib/api";

export default function HistoryTable() {
  const [games, setGames] = useState<Game[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getGames()
      .then(setGames)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-gray-500">Loading...</div>;

  if (games.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500 text-lg">No games played yet.</p>
        <a href="/" className="text-red-400 hover:text-red-300 mt-2 inline-block">Start a new game</a>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="py-3 px-4 text-gray-400 font-medium">Date</th>
            <th className="py-3 px-4 text-gray-400 font-medium">Agents</th>
            <th className="py-3 px-4 text-gray-400 font-medium">Werewolf</th>
            <th className="py-3 px-4 text-gray-400 font-medium">Winner</th>
            <th className="py-3 px-4 text-gray-400 font-medium">Deception Score</th>
            <th className="py-3 px-4 text-gray-400 font-medium">Status</th>
          </tr>
        </thead>
        <tbody>
          {games.map((game) => (
            <tr key={game.id} className="border-b border-gray-800 hover:bg-gray-900">
              <td className="py-3 px-4 text-sm">{new Date(game.created_at).toLocaleString()}</td>
              <td className="py-3 px-4">
                <div className="flex gap-1 flex-wrap">
                  {game.agents.map((a) => (
                    <span
                      key={a.id}
                      className="text-xs px-2 py-0.5 rounded capitalize"
                      style={{
                        backgroundColor: (AGENT_COLORS[a.agent_name] || "#6B7280") + "20",
                        color: AGENT_COLORS[a.agent_name] || "#9CA3AF",
                      }}
                    >
                      {a.agent_name}
                    </span>
                  ))}
                </div>
              </td>
              <td className="py-3 px-4 text-red-400 capitalize font-semibold">
                {game.werewolf_agent}
              </td>
              <td className="py-3 px-4">
                <span className={game.winner === "villagers" ? "text-green-400" : "text-red-400"}>
                  {game.winner || "-"}
                </span>
              </td>
              <td className="py-3 px-4">{game.deception_score} rounds</td>
              <td className="py-3 px-4">
                <a
                  href={`/game/${game.id}`}
                  className={`text-xs px-2 py-1 rounded ${
                    game.status === "finished"
                      ? "bg-gray-700 text-gray-300"
                      : "bg-green-900 text-green-300"
                  }`}
                >
                  {game.status === "finished" ? "View" : "Live"}
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
