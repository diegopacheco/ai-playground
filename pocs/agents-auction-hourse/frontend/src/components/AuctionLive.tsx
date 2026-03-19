import { useAuctionSSE } from "../hooks/useAuctionSSE.ts";
import { BidCard } from "./BidCard.tsx";
import { RoundSummary } from "./RoundSummary.tsx";
import { Leaderboard, FinalLeaderboard } from "./Leaderboard.tsx";
import { AGENT_COLORS } from "../types/index.ts";
import type { Agent } from "../types/index.ts";

interface AuctionLiveProps {
  auctionId: string;
  agents: Agent[];
}

export function AuctionLive({ auctionId, agents }: AuctionLiveProps) {
  const sse = useAuctionSSE(auctionId);

  if (sse.status === "connecting") {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-amber-400 text-xl animate-pulse">
          Connecting to auction...
        </div>
      </div>
    );
  }

  if (sse.status === "error") {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-400 text-xl">
          Error: {sse.errorMessage || "Connection lost"}
        </div>
      </div>
    );
  }

  if (sse.status === "finished" && sse.standings.length > 0) {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        <h2 className="text-3xl font-bold text-amber-400 text-center">
          Auction Complete
        </h2>
        <FinalLeaderboard standings={sse.standings} winner={sse.winner || ""} />
        <div className="space-y-3">
          <h3 className="text-xl font-bold text-gray-300">Round History</h3>
          {sse.rounds.map((r) => (
            <div key={r.round_number} className="space-y-2">
              <RoundSummary round={r} />
              <div className="pl-4 space-y-2">
                {r.bids.map((b) => (
                  <BidCard key={`${r.round_number}-${b.agent_name}`} bid={b} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-6">
        <div className="text-center">
          <span className="text-gray-400 text-sm">
            Round {sse.currentRound} / 3
          </span>
          <div className="flex justify-center gap-2 mt-1">
            {[1, 2, 3].map((r) => (
              <div
                key={r}
                className={`w-16 h-1 rounded ${
                  r <= sse.currentRound ? "bg-amber-400" : "bg-gray-700"
                }`}
              />
            ))}
          </div>
        </div>

        {sse.currentItem && (
          <div className="bg-gray-800 rounded-xl p-8 text-center border border-gray-700">
            <div className="text-6xl mb-4">{sse.currentItemEmoji}</div>
            <h2 className="text-2xl font-bold text-white">{sse.currentItem}</h2>
          </div>
        )}

        <div className="space-y-3">
          {sse.bids.map((bid, i) => (
            <BidCard key={`${bid.agent_name}-${i}`} bid={bid} isRevealing />
          ))}

          {sse.thinkingAgent && (
            <div
              className="bg-gray-800 rounded-lg p-4 border-l-4 animate-pulse"
              style={{
                borderLeftColor: AGENT_COLORS[sse.thinkingAgent] || "#6b7280",
              }}
            >
              <span
                className="font-bold capitalize"
                style={{
                  color: AGENT_COLORS[sse.thinkingAgent] || "#6b7280",
                }}
              >
                {sse.thinkingAgent}
              </span>
              <span className="text-gray-400 ml-2">is thinking...</span>
            </div>
          )}
        </div>

        {sse.status === "round_end" && sse.rounds.length > 0 && (
          <div className="bg-amber-900/20 border border-amber-600 rounded-lg p-4 text-center">
            <span className="text-amber-400 font-bold">
              {sse.rounds[sse.rounds.length - 1].winner_agent} wins this round
              with ${sse.rounds[sse.rounds.length - 1].winning_bid}!
            </span>
          </div>
        )}

        {sse.rounds.length > 0 && (
          <div className="space-y-2">
            <h3 className="text-gray-400 text-sm font-medium">
              Previous Rounds
            </h3>
            {sse.rounds.map((r) => (
              <RoundSummary key={r.round_number} round={r} />
            ))}
          </div>
        )}
      </div>

      <div className="lg:col-span-1">
        <Leaderboard agents={agents} rounds={sse.rounds} currentBids={sse.bids} />
      </div>
    </div>
  );
}
