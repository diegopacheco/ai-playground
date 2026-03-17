import { CreateQueryRequest, CreateQueryResponse, QueryRecord, TableInfo } from '../types'

const BASE = '/api'

export async function createQuery(req: CreateQueryRequest): Promise<CreateQueryResponse> {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  return res.json()
}

export async function getQueries(): Promise<QueryRecord[]> {
  const res = await fetch(`${BASE}/queries`)
  return res.json()
}

export async function getQuery(id: string): Promise<QueryRecord | null> {
  const res = await fetch(`${BASE}/queries/${id}`)
  return res.json()
}

export async function getSchema(): Promise<TableInfo[]> {
  const res = await fetch(`${BASE}/schema`)
  return res.json()
}

export function getStreamUrl(id: string): string {
  return `${BASE}/query/${id}/stream`
}
