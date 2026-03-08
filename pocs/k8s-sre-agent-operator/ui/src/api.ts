export interface ClusterObject {
  namespace: string
  kind: string
  name: string
  status: string
  ready: string
  restarts: number
  age: string
  yaml: string
}

export interface HistoryEvent {
  timestamp: string
  event_type: string
  summary: string
  details: string
  success: boolean
}

export interface FixResult {
  diagnostics: string
  claude_response: string
  kubectl_output: string
  success: boolean
}

export async function fetchStatus(): Promise<ClusterObject[]> {
  const res = await fetch('/api/status')
  return res.json()
}

export async function fetchLogs(): Promise<string> {
  const res = await fetch('/api/logs')
  return res.text()
}

export async function fetchHistory(): Promise<HistoryEvent[]> {
  const res = await fetch('/api/history')
  return res.json()
}

export async function postFix(): Promise<FixResult> {
  const res = await fetch('/api/fix', { method: 'POST' })
  return res.json()
}
