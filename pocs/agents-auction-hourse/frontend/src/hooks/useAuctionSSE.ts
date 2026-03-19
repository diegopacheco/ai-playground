import { useState, useEffect, useCallback } from "react";
import type { Bid, Round, AuctionAgent } from "../types/index.ts";

interface SSEState {
  status: "connecting" | "bidding" | "round_end" | "finished" | "error";
  currentRound: number;
  currentItem: string;
  currentItemEmoji: string;
  thinkingAgent: string | null;
  bids: Bid[];
  rounds: Round[];
  standings: AuctionAgent[];
  winner: string | null;
  errorMessage: string | null;
}

const initialState: SSEState = {
  status: "connecting",
  currentRound: 0,
  currentItem: "",
  currentItemEmoji: "",
  thinkingAgent: null,
  bids: [],
  rounds: [],
  standings: [],
  winner: null,
  errorMessage: null,
};

export function useAuctionSSE(auctionId: string | null) {
  const [state, setState] = useState<SSEState>(initialState);

  const reset = useCallback(() => {
    setState(initialState);
  }, []);

  useEffect(() => {
    if (!auctionId) return;

    const eventSource = new EventSource(
      `/api/auctions/${auctionId}/stream`
    );

    eventSource.addEventListener("round_start", (e) => {
      const data = JSON.parse(e.data);
      setState((prev) => ({
        ...prev,
        status: "bidding",
        currentRound: data.round,
        currentItem: data.item,
        currentItemEmoji: data.item_emoji,
        thinkingAgent: null,
        bids: [],
      }));
    });

    eventSource.addEventListener("agent_thinking", (e) => {
      const data = JSON.parse(e.data);
      setState((prev) => ({
        ...prev,
        thinkingAgent: data.agent,
      }));
    });

    eventSource.addEventListener("agent_bid", (e) => {
      const data = JSON.parse(e.data);
      const bid: Bid = {
        agent_name: data.agent,
        amount: data.bid,
        reasoning: data.reasoning,
        fallback: data.fallback || false,
        response_time_ms: data.response_time_ms || 0,
      };
      setState((prev) => ({
        ...prev,
        thinkingAgent: null,
        bids: [...prev.bids, bid],
      }));
    });

    eventSource.addEventListener("round_result", (e) => {
      const data = JSON.parse(e.data);
      setState((prev) => {
        const round: Round = {
          round_number: data.round,
          item_name: prev.currentItem,
          item_emoji: prev.currentItemEmoji,
          winner_agent: data.winner,
          winning_bid: data.winning_bid,
          bids: [...prev.bids],
        };
        return {
          ...prev,
          status: "round_end",
          rounds: [...prev.rounds, round],
        };
      });
    });

    eventSource.addEventListener("auction_over", (e) => {
      const data = JSON.parse(e.data);
      setState((prev) => ({
        ...prev,
        status: "finished",
        winner: data.winner,
        standings: data.final_standings || [],
      }));
      eventSource.close();
    });

    eventSource.addEventListener("error", (e) => {
      if (e instanceof MessageEvent) {
        const data = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          status: "error",
          errorMessage: data.message,
        }));
      }
      eventSource.close();
    });

    return () => {
      eventSource.close();
    };
  }, [auctionId]);

  return { ...state, reset };
}
