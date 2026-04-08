import { GameConfig, ScoreEntry } from "./types";

const BASE = "http://localhost:8080";

export async function fetchScores(): Promise<ScoreEntry[]> {
  const res = await fetch(`${BASE}/api/scores`);
  return res.json();
}

export async function submitScore(score: Omit<ScoreEntry, "id" | "created_at">): Promise<ScoreEntry> {
  const res = await fetch(`${BASE}/api/scores`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(score),
  });
  return res.json();
}

export async function fetchConfig(): Promise<GameConfig> {
  const res = await fetch(`${BASE}/api/config`);
  return res.json();
}

export async function updateConfig(config: GameConfig): Promise<GameConfig> {
  const res = await fetch(`${BASE}/api/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  return res.json();
}
