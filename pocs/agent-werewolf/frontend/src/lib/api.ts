import { AgentInfo, AgentSelection, Game } from "@/types";

const API_BASE = "http://localhost:3000/api";

export async function createGame(agents: AgentSelection[]): Promise<{ id: string }> {
  const res = await fetch(`${API_BASE}/games`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agents }),
  });
  return res.json();
}

export async function getGames(): Promise<Game[]> {
  const res = await fetch(`${API_BASE}/games`);
  return res.json();
}

export async function getGame(id: string): Promise<Game> {
  const res = await fetch(`${API_BASE}/games/${id}`);
  return res.json();
}

export async function getAgents(): Promise<AgentInfo[]> {
  const res = await fetch(`${API_BASE}/agents`);
  return res.json();
}

export function getStreamUrl(gameId: string): string {
  return `${API_BASE}/games/${gameId}/stream`;
}
