export interface MatchRecord {
  id: string
  agent_a: string
  agent_b: string
  winner: string | null
  is_draw: boolean
  moves: string
  started_at: string
  ended_at: string | null
  duration_ms: number | null
}

export interface BoardUpdateEvent {
  type: 'board_update'
  board: string[][]
  current_player: string
  last_move: number | null
}

export interface AgentThinkingEvent {
  type: 'agent_thinking'
  agent: string
}

export interface AgentMovedEvent {
  type: 'agent_moved'
  agent: string
  column: number
}

export interface GameOverEvent {
  type: 'game_over'
  winner: string | null
  is_draw: boolean
  duration_ms: number
}

export interface ErrorEvent {
  type: 'error'
  message: string
}

export type GameEvent = BoardUpdateEvent | AgentThinkingEvent | AgentMovedEvent | GameOverEvent | ErrorEvent

export interface AgentInfo {
  name: string
  model: string
  color: string
}

export const AGENT_INFO: Record<string, AgentInfo> = {
  claude: { name: 'Claude', model: 'opus-4-5', color: '#D97706' },
  gemini: { name: 'Gemini', model: 'gemini-3', color: '#4285F4' },
  copilot: { name: 'Copilot', model: 'claude-sonnet-4', color: '#6366F1' },
  codex: { name: 'Codex', model: 'gpt-5.2', color: '#10B981' },
}

export function getAgentInfo(agent: string): AgentInfo {
  return AGENT_INFO[agent.toLowerCase()] || { name: agent, model: 'unknown', color: '#6B7280' }
}
