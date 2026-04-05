import { Agent, CreateAgentRequest } from '../types'

const BASE = '/api'

export async function spawnAgent(req: CreateAgentRequest): Promise<Agent> {
  const res = await fetch(`${BASE}/agents/spawn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  return res.json()
}

export async function fetchAgents(): Promise<Agent[]> {
  const res = await fetch(`${BASE}/agents`)
  return res.json()
}

export async function fetchAgent(id: string): Promise<Agent & { messages: any[] }> {
  const res = await fetch(`${BASE}/agents/${id}`)
  return res.json()
}

export async function stopAgent(id: string): Promise<void> {
  await fetch(`${BASE}/agents/${id}`, { method: 'DELETE' })
}

export function getStreamUrl(id: string): string {
  return `${BASE}/agents/${id}/stream`
}

export async function clearAllAgents(): Promise<void> {
  await fetch(`${BASE}/agents/clear`, { method: 'DELETE' })
}

export async function chatWithAgent(id: string, message: string): Promise<{ role: string; content: string }> {
  const res = await fetch(`${BASE}/agents/${id}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })
  return res.json()
}
