import type { TaskRequest, TaskResponse, TaskStatus, ProjectInfo, ProjectDetail, ConfigRequest, ConfigResponse } from '../types'

const API_BASE = '/api'

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Request failed' }))
    throw new Error(err.error || 'Request failed')
  }
  return res.json()
}

export async function getProjects(): Promise<ProjectInfo[]> {
  return fetchJson('/projects')
}

export async function getProject(name: string): Promise<ProjectDetail> {
  return fetchJson(`/projects/${encodeURIComponent(name)}`)
}

export async function submitTask(req: TaskRequest): Promise<TaskResponse> {
  return fetchJson('/tasks', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

export async function getTaskStatus(taskId: string): Promise<TaskStatus> {
  return fetchJson(`/tasks/${taskId}/status`)
}

export async function getConfig(): Promise<ConfigResponse> {
  return fetchJson('/config')
}

export async function updateConfig(req: ConfigRequest): Promise<ConfigResponse> {
  return fetchJson('/config', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}
