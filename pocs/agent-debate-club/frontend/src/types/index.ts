export interface AgentInfo {
  id: string;
  name: string;
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

export interface CreateDebateRequest {
  topic: string;
  agent_a: string;
  agent_b: string;
  agent_judge: string;
  duration_seconds: number;
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
