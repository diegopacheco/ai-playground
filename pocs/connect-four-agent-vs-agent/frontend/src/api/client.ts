const API_BASE = '/api'

export async function fetchAgents(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/agents`)
  return response.json()
}

export async function startGame(agentA: string, agentB: string): Promise<{ game_id: string }> {
  const response = await fetch(`${API_BASE}/game/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ agent_a: agentA, agent_b: agentB }),
  })
  return response.json()
}

export async function fetchHistory(): Promise<import('../types').MatchRecord[]> {
  const response = await fetch(`${API_BASE}/history`)
  return response.json()
}

export async function fetchMatch(id: string): Promise<import('../types').MatchRecord | null> {
  const response = await fetch(`${API_BASE}/history/${id}`)
  return response.json()
}

export function createGameStream(gameId: string): EventSource {
  return new EventSource(`${API_BASE}/game/${gameId}/stream`)
}
