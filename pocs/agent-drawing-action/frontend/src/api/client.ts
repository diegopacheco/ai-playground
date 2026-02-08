const BASE = '/api'

export interface EngineInfo {
  id: string
  name: string
}

export interface GuessRecord {
  id: string
  engine: string
  guess: string | null
  status: string
  created_at: string
  completed_at: string | null
  error: string | null
}

export interface CreateGuessRequest {
  engine: string
  image: string
}

export interface CreateGuessResponse {
  id: string
  guess: string | null
  engine: string
  status: string
  error: string | null
}

export async function fetchEngines(): Promise<EngineInfo[]> {
  const res = await fetch(`${BASE}/engines`)
  return res.json()
}

export async function createGuess(req: CreateGuessRequest): Promise<CreateGuessResponse> {
  const res = await fetch(`${BASE}/guesses`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  return res.json()
}

export async function fetchGuesses(): Promise<GuessRecord[]> {
  const res = await fetch(`${BASE}/guesses`)
  return res.json()
}

export async function deleteGuess(id: string): Promise<boolean> {
  const res = await fetch(`${BASE}/guesses/${id}`, { method: 'DELETE' })
  return res.json()
}

export function drawingUrl(id: string): string {
  return `/output/${id}/drawing.png`
}
