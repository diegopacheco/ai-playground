export interface Game {
  id: string;
  status: string;
  winner: string | null;
  werewolf_agent: string | null;
  deception_score: number;
  created_at: string;
  ended_at: string | null;
  agents: GameAgent[];
  rounds: Round[];
}

export interface GameAgent {
  id: string;
  game_id: string;
  agent_name: string;
  model: string;
  role: string;
  alive: boolean;
  votes_correct: number;
  votes_total: number;
}

export interface Round {
  id: string;
  game_id: string;
  round_number: number;
  phase: string;
  eliminated_agent: string | null;
  eliminated_by: string | null;
  messages: Message[];
}

export interface Message {
  id: string;
  round_id: string;
  agent_name: string;
  message_type: string;
  content: string;
  target: string | null;
  raw_output: string | null;
  response_time_ms: number | null;
  created_at: string;
}

export interface AgentInfo {
  name: string;
  models: string[];
  default_model: string;
}

export interface AgentSelection {
  name: string;
  model: string;
}

export const AGENT_COLORS: Record<string, string> = {
  claude: "#D97706",
  gemini: "#2563EB",
  copilot: "#7C3AED",
  codex: "#059669",
};

export function getAgentColor(name: string): string {
  const base = name.replace(/-\d+$/, "");
  return AGENT_COLORS[base] || "#6B7280";
}
