import { TraceRecord, RunResponse } from '../types'

const API = 'http://localhost:3000/api'

export async function runAgent(topic: string, agent: string): Promise<RunResponse> {
  const res = await fetch(`${API}/agent/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic, agent }),
  })
  return res.json()
}

export async function getTraces(): Promise<TraceRecord[]> {
  const res = await fetch(`${API}/traces`)
  return res.json()
}

export async function getTrace(traceId: string): Promise<TraceRecord | null> {
  const res = await fetch(`${API}/traces/${traceId}`)
  return res.json()
}

export async function getAgents(): Promise<[string, string][]> {
  const res = await fetch(`${API}/agents`)
  return res.json()
}
