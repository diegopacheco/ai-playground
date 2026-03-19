export interface AgentInfo {
  name: string;
  models: string[];
}

export interface Game {
  id: string;
  terminatorAgent: string;
  terminatorModel: string;
  mosquitoAgent: string;
  mosquitoModel: string;
  gridSize: number;
  winner: string | null;
  totalCycles: number;
  maxMosquitos: number;
  totalKills: number;
  totalHatched: number;
  totalDates: number;
  status: string;
  createdAt: string;
  endedAt: string | null;
}

export interface MosquitoState {
  id: string;
  x: number;
  y: number;
  age: number;
}

export interface EggState {
  id: string;
  x: number;
  y: number;
  ticks: number;
}

export interface CycleUpdate {
  cycle: number;
  terminator: { x: number; y: number };
  mosquitos: MosquitoState[];
  eggs: EggState[];
  kills: KillEvent[];
  dates: DateEvent[];
  hatches: HatchEvent[];
  deaths: string[];
  total_kills: number;
  total_hatched: number;
  total_dates: number;
  alive_count: number;
  egg_count: number;
}

export interface KillEvent {
  position: { x: number; y: number };
  killed_mosquitos: string[];
  killed_eggs: string[];
}

export interface DateEvent {
  position: { x: number; y: number };
  mosquito_ids: string[];
  egg_id: string;
}

export interface HatchEvent {
  position: { x: number; y: number };
  egg_id: string;
  new_mosquito_id: string;
}

export interface GameStartEvent {
  grid_size: number;
  terminator: { x: number; y: number };
  mosquitos: { id: string; x: number; y: number }[];
  terminator_agent: string;
  mosquito_agent: string;
}

export interface GameOverEvent {
  winner: string;
  cycles: number;
  total_kills: number;
  total_hatched: number;
  total_dates: number;
  max_mosquitos: number;
  alive_mosquitos: number;
  active_eggs: number;
}

export const AVAILABLE_AGENTS: AgentInfo[] = [
  { name: "claude", models: ["opus", "sonnet", "haiku"] },
  { name: "gemini", models: ["gemini-3.1-pro", "gemini-3-flash", "gemini-2.5-pro"] },
  { name: "copilot", models: ["claude-sonnet-4.6", "claude-sonnet-4.5", "gemini-3-pro"] },
  { name: "codex", models: ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"] },
];

export const AGENT_COLORS: Record<string, string> = {
  claude: "#a855f7",
  gemini: "#3b82f6",
  copilot: "#22c55e",
  codex: "#f97316",
};
