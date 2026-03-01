import type { Player, Game, FlipResponse, LeaderboardEntry } from "./types"

const BASE_URL = "http://localhost:3000"

export async function createPlayer(name: string): Promise<Player> {
  const res = await fetch(`${BASE_URL}/api/players`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  })
  if (!res.ok) throw new Error("Failed to create player")
  return res.json()
}

export async function createGame(player_id: number): Promise<Game> {
  const res = await fetch(`${BASE_URL}/api/games`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ player_id }),
  })
  if (!res.ok) throw new Error("Failed to create game")
  return res.json()
}

export async function getGame(id: number): Promise<Game> {
  const res = await fetch(`${BASE_URL}/api/games/${id}`)
  if (!res.ok) throw new Error("Failed to get game")
  return res.json()
}

export async function flipCard(gameId: number, position: number): Promise<FlipResponse> {
  const res = await fetch(`${BASE_URL}/api/games/${gameId}/flip`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ position }),
  })
  if (!res.ok) throw new Error("Failed to flip card")
  return res.json()
}

export async function getLeaderboard(): Promise<LeaderboardEntry[]> {
  const res = await fetch(`${BASE_URL}/api/leaderboard`)
  if (!res.ok) throw new Error("Failed to get leaderboard")
  return res.json()
}
