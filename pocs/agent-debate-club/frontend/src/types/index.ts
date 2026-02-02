export interface AgentInfo {
  id: string;
  name: string;
  model: string;
  color: string;
}

export const AGENT_INFO: Record<string, AgentInfo> = {
  claude: { id: 'claude', name: 'Claude', model: 'opus-4-5', color: '#D97706' },
  gemini: { id: 'gemini', name: 'Gemini', model: 'gemini-3.0', color: '#4285F4' },
  copilot: { id: 'copilot', name: 'Copilot', model: 'claude-sonnet-4', color: '#6366F1' },
  codex: { id: 'codex', name: 'Codex', model: 'gpt-5.2-codex', color: '#10B981' },
};

export function getAgentInfo(agentId: string): AgentInfo {
  return AGENT_INFO[agentId.toLowerCase()] || { id: agentId, name: agentId, model: 'unknown', color: '#6B7280' };
}

export interface Message {
  id: number;
  debate_id: string;
  agent: string;
  content: string;
  stance: string;
  created_at: string;
}

export interface Debate {
  id: string;
  topic: string;
  agent_a: string;
  agent_b: string;
  agent_judge: string;
  winner: string | null;
  judge_reason: string | null;
  duration_seconds: number;
  started_at: string;
  ended_at: string | null;
}

export interface DebateWithMessages extends Debate {
  messages: Message[];
}

export type DebateStyle = 'neutral' | 'ArthurSchopenhauer' | 'ExtremeRadical' | 'Zen' | 'Idiocracy' | 'comedian' | 'gangster' | 'political_candidate';

export const DEBATE_STYLES: DebateStyle[] = ['neutral', 'ArthurSchopenhauer', 'ExtremeRadical', 'Zen', 'Idiocracy', 'comedian', 'gangster', 'political_candidate'];

export interface CreateDebateRequest {
  topic: string;
  agent_a: string;
  agent_b: string;
  agent_judge: string;
  duration_seconds: number;
  style_a: DebateStyle;
  style_b: DebateStyle;
}

export interface CreateDebateResponse {
  id: string;
}

export interface DebateResult {
  winner: string;
  reason: string;
  duration_ms: number;
}

export type DebateStatus = 'idle' | 'thinking' | 'over';

export interface AgentThinkingEvent {
  type: 'agent_thinking';
  agent: string;
}

export interface AgentMessageEvent {
  type: 'agent_message';
  agent: string;
  content: string;
  stance: string;
}

export interface DebateOverEvent {
  type: 'debate_over';
  winner: string;
  reason: string;
  duration_ms: number;
}

export interface ErrorEvent {
  type: 'error';
  message: string;
}

export type DebateEvent = AgentThinkingEvent | AgentMessageEvent | DebateOverEvent | ErrorEvent;
