const BASE = 'http://localhost:3000/api'

export async function createGame(playerName: string, setting: string): Promise<string> {
  const res = await fetch(`${BASE}/games`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ player_name: playerName, setting }),
  })
  const data = await res.json()
  return data.id
}

export async function sendAction(gameId: string, action: string): Promise<void> {
  await fetch(`${BASE}/games/${gameId}/action`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  })
}

export async function getGame(gameId: string) {
  const res = await fetch(`${BASE}/games/${gameId}`)
  return res.json()
}

export async function listGames() {
  const res = await fetch(`${BASE}/games`)
  return res.json()
}

export function streamUrl(gameId: string): string {
  return `${BASE}/games/${gameId}/stream`
}
