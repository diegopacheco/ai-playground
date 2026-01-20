export interface Agent {
  name: string
  model: string
  status: 'pending' | 'running' | 'done' | 'error' | 'timeout'
  worktree: string
}

export interface RunResponse {
  session_id: string
}

export interface StatusResponse {
  agents: Agent[]
}

export interface FileEntry {
  path: string
  name: string
  is_dir: boolean
}

export interface FilesResponse {
  files: FileEntry[]
}

export interface FileContentResponse {
  content: string
  language: string
}

const BASE_URL = '/api'

export async function runAgents(prompt: string, projectName: string): Promise<RunResponse> {
  const response = await fetch(`${BASE_URL}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, project_name: projectName }),
  })
  if (!response.ok) throw new Error('Failed to start agents')
  return response.json()
}

export async function getStatus(sessionId: string): Promise<StatusResponse> {
  const response = await fetch(`${BASE_URL}/status/${sessionId}`)
  if (!response.ok) throw new Error('Failed to get status')
  return response.json()
}

export async function getFiles(sessionId: string, agentName: string): Promise<FilesResponse> {
  const response = await fetch(`${BASE_URL}/files/${sessionId}/${agentName}`)
  if (!response.ok) throw new Error('Failed to get files')
  return response.json()
}

export async function getFileContent(sessionId: string, agentName: string, filePath: string): Promise<FileContentResponse> {
  const response = await fetch(`${BASE_URL}/file/${sessionId}/${agentName}?path=${encodeURIComponent(filePath)}`)
  if (!response.ok) throw new Error('Failed to get file content')
  return response.json()
}
