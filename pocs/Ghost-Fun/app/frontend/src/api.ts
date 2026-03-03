const BASE = 'http://localhost:8080/api'

export interface Score {
  id: number
  player_name: string
  moves: number
  time_taken: number
  created_at: string
}

export async function fetchLeaderboard(): Promise<Score[]> {
  const res = await fetch(`${BASE}/leaderboard`)
  return res.json()
}

export async function submitScore(player_name: string, moves: number, time_taken: number): Promise<void> {
  await fetch(`${BASE}/scores`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ player_name, moves, time_taken }),
  })
}
