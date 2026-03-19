import { useEffect, useState, useCallback } from "react";
import { listGames } from "~/api/game";
import type { Game } from "~/types";

export default function HistoryTable() {
  const [games, setGames] = useState<Game[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchGames = useCallback(() => {
    setLoading(true);
    fetch("/api/games")
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) {
          setGames(data);
        }
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchGames();
  }, [fetchGames]);

  if (loading) {
    return <div className="text-center text-gray-500 py-10">Loading...</div>;
  }

  if (games.length === 0) {
    return (
      <div className="text-center text-gray-500 py-10">
        No games played yet. Start a simulation!
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-8">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-300">Game History</h2>
        <button
          onClick={fetchGames}
          className="bg-emerald-700 hover:bg-emerald-600 text-white px-4 py-2 rounded-lg text-sm transition-colors"
        >
          Refresh
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500">
              <th className="text-left py-3 px-4">Date</th>
              <th className="text-left py-3 px-4">Terminator</th>
              <th className="text-left py-3 px-4">Mosquitos</th>
              <th className="text-left py-3 px-4">Winner</th>
              <th className="text-right py-3 px-4">Cycles</th>
              <th className="text-right py-3 px-4">Max Mosquitos</th>
              <th className="text-right py-3 px-4">Total Kills</th>
            </tr>
          </thead>
          <tbody>
            {games.map((game) => (
              <tr
                key={game.id}
                className="border-b border-gray-800/50 hover:bg-gray-900/50"
              >
                <td className="py-3 px-4 text-gray-400">
                  {new Date(game.createdAt).toLocaleDateString()}
                </td>
                <td className="py-3 px-4">
                  <span className="text-white">{game.terminatorAgent}</span>
                  <span className="text-gray-600 text-xs ml-1">
                    {game.terminatorModel}
                  </span>
                </td>
                <td className="py-3 px-4">
                  <span className="text-green-400">{game.mosquitoAgent}</span>
                  <span className="text-gray-600 text-xs ml-1">
                    {game.mosquitoModel}
                  </span>
                </td>
                <td className="py-3 px-4">
                  <span
                    className={`font-bold ${
                      game.winner === "terminator"
                        ? "text-white"
                        : game.winner === "mosquitos"
                          ? "text-green-400"
                          : "text-yellow-400"
                    }`}
                  >
                    {game.winner || "running"}
                  </span>
                </td>
                <td className="py-3 px-4 text-right text-gray-300 font-mono">
                  {game.totalCycles}
                </td>
                <td className="py-3 px-4 text-right text-gray-300 font-mono">
                  {game.maxMosquitos}
                </td>
                <td className="py-3 px-4 text-right text-gray-300 font-mono">
                  {game.totalKills}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
