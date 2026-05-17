const BASE = "http://localhost:8080";

export type ApiCard = { id: number; symbol: string };
export type ApiScore = { name: string; moves: number; seconds: number };

export async function newGame(): Promise<ApiCard[]> {
  const res = await fetch(`${BASE}/api/games`, { method: "POST" });
  if (!res.ok) throw new Error(`new game failed: ${res.status}`);
  const data = await res.json();
  return data.deck;
}

export async function getScores(): Promise<ApiScore[]> {
  const res = await fetch(`${BASE}/api/scores`);
  if (!res.ok) throw new Error(`get scores failed: ${res.status}`);
  const data = await res.json();
  return data.scores;
}

export async function submitScore(score: ApiScore): Promise<void> {
  const res = await fetch(`${BASE}/api/scores`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(score),
  });
  if (!res.ok) throw new Error(`submit score failed: ${res.status}`);
}
