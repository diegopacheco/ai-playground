export interface AgentType {
  id: string
  name: string
  model: string
  color: string
}

export interface Agent {
  id: string
  name: string
  agent_type: string
  task: string
  status: 'spawning' | 'thinking' | 'working' | 'done' | 'error' | 'stopped'
  desk_index: number
  created_at: string
  completed_at: string | null
}

export interface AgentMessage {
  id: string
  agent_id: string
  content: string
  role: string
  created_at: string
}

export interface CreateAgentRequest {
  name: string
  agent_type: string
  task: string
}

export interface SSEEvent {
  type: 'agent_status' | 'agent_message' | 'agent_done' | 'agent_error'
  agent_id: string
  status?: string
  content?: string
  message?: string
}

export const AGENT_TYPES: AgentType[] = [
  { id: 'claude', name: 'Claude', model: 'opus', color: '#D97706' },
  { id: 'gemini', name: 'Gemini', model: 'gemini-3.0', color: '#4285F4' },
  { id: 'copilot', name: 'Copilot', model: 'claude-sonnet-4', color: '#6366F1' },
  { id: 'codex', name: 'Codex', model: 'gpt-5.4', color: '#10B981' },
]

export const CHAR_FRAME_W = 16
export const CHAR_FRAME_H = 32
export const CHAR_COLS = 7
export const CHAR_DIRECTIONS = 3
export const TILE_SIZE = 16
export const SCALE = 3
export const DISPLAY_TILE = TILE_SIZE * SCALE
