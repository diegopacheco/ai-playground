import type { Round } from "../types/index.ts";
import { AGENT_COLORS } from "../types/index.ts";

interface RoundSummaryProps {
  round: Round;
}

export function RoundSummary({ round }: RoundSummaryProps) {
  const winnerColor = AGENT_COLORS[round.winner_agent] || "#6b7280";

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-400 text-sm">Round {round.round_number}</span>
        <span className="text-2xl">
          {round.item_emoji} {round.item_name}
        </span>
      </div>
      <div className="flex items-center gap-2 mt-2">
        <span className="text-gray-400">Winner:</span>
        <span className="font-bold capitalize" style={{ color: winnerColor }}>
          {round.winner_agent}
        </span>
        <span className="text-amber-400 font-bold">${round.winning_bid}</span>
      </div>
    </div>
  );
}
