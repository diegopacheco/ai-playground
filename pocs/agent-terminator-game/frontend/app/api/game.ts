import type { Game } from "~/types";

export async function createGame(params: {
  terminator_agent: string;
  terminator_model: string;
  mosquito_agent: string;
  mosquito_model: string;
  grid_size: string;
}): Promise<{ id: string; status: string }> {
  const res = await fetch("/api/games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function listGames(): Promise<Game[]> {
  const res = await fetch("/api/games");
  return res.json();
}

export async function getGame(id: string): Promise<Game> {
  const res = await fetch(`/api/games/${id}`);
  return res.json();
}

export async function stopGame(id: string): Promise<void> {
  await fetch(`/api/games/${id}/stop`, { method: "POST" });
}
