import type { QueryRequest, QueryResponse } from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8080'

export async function sendQuery(req: QueryRequest): Promise<QueryResponse> {
  const resp = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!resp.ok) {
    let message = `request failed (${resp.status})`
    try {
      const data = await resp.json()
      if (data?.error) message = data.error
    } catch {
      message = `request failed (${resp.status})`
    }
    throw new Error(message)
  }
  return resp.json()
}
