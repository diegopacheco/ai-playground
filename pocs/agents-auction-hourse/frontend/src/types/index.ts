export interface Agent {
  name: string;
  model: string;
  budget: number;
}

export interface Auction {
  id: string;
  agents: AuctionAgent[];
  rounds: Round[];
  winner: string;
  created_at: string;
  ended_at: string;
}

export interface AuctionAgent {
  agent_name: string;
  model: string;
  initial_budget: number;
  remaining_budget: number;
  items_won: number;
}

export interface Round {
  round_number: number;
  item_name: string;
  item_emoji: string;
  winner_agent: string;
  winning_bid: number;
  bids: Bid[];
}

export interface Bid {
  agent_name: string;
  amount: number;
  reasoning: string;
  fallback: boolean;
  response_time_ms: number;
}

export interface AgentOption {
  name: string;
  models: string[];
  defaultModel: string;
  color: string;
}

export const AVAILABLE_AGENTS: AgentOption[] = [
  {
    name: "claude",
    models: ["opus", "sonnet", "haiku"],
    defaultModel: "sonnet",
    color: "#a855f7",
  },
  {
    name: "gemini",
    models: ["gemini-3.1-pro", "gemini-3-flash", "gemini-2.5-pro"],
    defaultModel: "gemini-3.1-pro",
    color: "#3b82f6",
  },
  {
    name: "copilot",
    models: ["claude-sonnet-4.6", "claude-sonnet-4.5", "gemini-3-pro"],
    defaultModel: "claude-sonnet-4.6",
    color: "#22c55e",
  },
  {
    name: "codex",
    models: ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"],
    defaultModel: "gpt-5.4",
    color: "#f97316",
  },
];

export const AGENT_COLORS: Record<string, string> = {
  claude: "#a855f7",
  gemini: "#3b82f6",
  copilot: "#22c55e",
  codex: "#f97316",
};
