import type { Bid } from "../types/index.ts";
import { AGENT_COLORS } from "../types/index.ts";

interface BidCardProps {
  bid: Bid;
  isRevealing?: boolean;
}

export function BidCard({ bid, isRevealing }: BidCardProps) {
  const color = AGENT_COLORS[bid.agent_name] || "#6b7280";

  return (
    <div
      className={`bg-gray-800 rounded-lg p-4 border-l-4 transition-all duration-500 ${
        isRevealing ? "animate-slide-in opacity-100" : "opacity-100"
      }`}
      style={{ borderLeftColor: color }}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="font-bold text-lg capitalize" style={{ color }}>
          {bid.agent_name}
        </span>
        <div className="flex items-center gap-2">
          {bid.fallback && (
            <span className="bg-yellow-600 text-yellow-100 text-xs px-2 py-1 rounded font-medium">
              FALLBACK
            </span>
          )}
          <span className="text-2xl font-bold text-amber-400">
            ${bid.amount}
          </span>
        </div>
      </div>
      <p className="text-gray-300 text-sm italic">"{bid.reasoning}"</p>
      {bid.response_time_ms > 0 && (
        <p className="text-gray-500 text-xs mt-1">
          {(bid.response_time_ms / 1000).toFixed(1)}s response time
        </p>
      )}
    </div>
  );
}
