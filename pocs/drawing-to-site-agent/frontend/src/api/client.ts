const BASE = '/api'

export interface EngineInfo {
  id: string
  name: string
}

export interface ProjectRecord {
  id: string
  name: string
  engine: string
  status: string
  created_at: string
  completed_at: string | null
  error: string | null
}

export interface CreateProjectRequest {
  name: string
  engine: string
  image: string
}

export interface CreateProjectResponse {
  id: string
}

export async function fetchEngines(): Promise<EngineInfo[]> {
  const res = await fetch(`${BASE}/engines`)
  return res.json()
}

export async function createProject(req: CreateProjectRequest): Promise<CreateProjectResponse> {
  const res = await fetch(`${BASE}/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  return res.json()
}

export async function fetchProjects(): Promise<ProjectRecord[]> {
  const res = await fetch(`${BASE}/projects`)
  return res.json()
}

export async function fetchProject(id: string): Promise<ProjectRecord | null> {
  const res = await fetch(`${BASE}/projects/${id}`)
  return res.json()
}

export async function deleteProject(id: string): Promise<boolean> {
  const res = await fetch(`${BASE}/projects/${id}`, { method: 'DELETE' })
  return res.json()
}

export function projectStreamUrl(id: string): string {
  return `${BASE}/projects/${id}/stream`
}

export function previewUrl(id: string): string {
  return `/output/${id}/index.html`
}
