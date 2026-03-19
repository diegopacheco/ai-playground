import type { AuctionAgent } from "../types/index.ts";
import { AGENT_COLORS } from "../types/index.ts";
import type { Bid, Round } from "../types/index.ts";

interface LeaderboardProps {
  agents: { name: string; model: string; budget: number }[];
  rounds: Round[];
  currentBids: Bid[];
}

interface Standing {
  name: string;
  itemsWon: number;
  totalSpent: number;
  remaining: number;
  initialBudget: number;
}

export function Leaderboard({ agents, rounds, currentBids }: LeaderboardProps) {
  const standings: Standing[] = agents.map((a) => {
    let totalSpent = 0;
    let itemsWon = 0;
    for (const r of rounds) {
      if (r.winner_agent === a.name) {
        itemsWon++;
        totalSpent += r.winning_bid;
      }
    }
    return {
      name: a.name,
      itemsWon,
      totalSpent,
      remaining: a.budget - totalSpent,
      initialBudget: a.budget,
    };
  });

  standings.sort((a, b) => {
    if (b.itemsWon !== a.itemsWon) return b.itemsWon - a.itemsWon;
    return a.totalSpent - b.totalSpent;
  });

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-amber-400 font-bold text-lg mb-4">Leaderboard</h3>
      <div className="space-y-4">
        {standings.map((s) => {
          const color = AGENT_COLORS[s.name] || "#6b7280";
          const budgetPercent = (s.remaining / s.initialBudget) * 100;
          return (
            <div key={s.name} className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-bold capitalize" style={{ color }}>
                  {s.name}
                </span>
                <span className="text-gray-400 text-sm">
                  {s.itemsWon} won | ${s.totalSpent} spent
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div
                  className="h-3 rounded-full transition-all duration-500"
                  style={{
                    width: `${Math.max(budgetPercent, 0)}%`,
                    backgroundColor: color,
                  }}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-500">
                <span>${s.remaining} remaining</span>
                <span>${s.initialBudget} initial</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface FinalLeaderboardProps {
  standings: AuctionAgent[];
  winner: string;
}

export function FinalLeaderboard({ standings, winner }: FinalLeaderboardProps) {
  const sorted = [...standings].sort((a, b) => {
    if (b.items_won !== a.items_won) return b.items_won - a.items_won;
    const aSpent = a.initial_budget - a.remaining_budget;
    const bSpent = b.initial_budget - b.remaining_budget;
    return aSpent - bSpent;
  });

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-amber-600">
      <h3 className="text-amber-400 font-bold text-xl mb-4 text-center">
        Final Results
      </h3>
      <div className="space-y-4">
        {sorted.map((s) => {
          const color = AGENT_COLORS[s.agent_name] || "#6b7280";
          const spent = s.initial_budget - s.remaining_budget;
          const isWinner = s.agent_name === winner;
          return (
            <div
              key={s.agent_name}
              className={`p-4 rounded-lg ${
                isWinner ? "bg-amber-900/30 border border-amber-600" : "bg-gray-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {isWinner && (
                    <span className="text-amber-400 font-bold text-sm bg-amber-900/50 px-2 py-1 rounded">
                      SAVVIEST BIDDER
                    </span>
                  )}
                  <span className="font-bold text-lg capitalize" style={{ color }}>
                    {s.agent_name}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-amber-400 font-bold">
                    {s.items_won} items won
                  </div>
                  <div className="text-gray-400 text-sm">${spent} spent</div>
                </div>
              </div>
              <div className="w-full bg-gray-600 rounded-full h-2 mt-2">
                <div
                  className="h-2 rounded-full"
                  style={{
                    width: `${(s.remaining_budget / s.initial_budget) * 100}%`,
                    backgroundColor: color,
                  }}
                />
              </div>
              <div className="text-xs text-gray-500 mt-1">
                ${s.remaining_budget} / ${s.initial_budget} remaining
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
