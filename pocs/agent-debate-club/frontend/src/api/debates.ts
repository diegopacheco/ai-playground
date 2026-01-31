import type { AgentInfo, CreateDebateRequest, CreateDebateResponse, Debate, DebateWithMessages } from '../types';

const API_BASE = 'http://localhost:3000/api';

export async function createDebate(req: CreateDebateRequest): Promise<CreateDebateResponse> {
  const response = await fetch(`${API_BASE}/debates`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return response.json();
}

export async function getDebates(): Promise<Debate[]> {
  const response = await fetch(`${API_BASE}/debates`);
  return response.json();
}

export async function getDebate(id: string): Promise<DebateWithMessages | null> {
  const response = await fetch(`${API_BASE}/debates/${id}`);
  return response.json();
}

export async function getAgents(): Promise<AgentInfo[]> {
  const response = await fetch(`${API_BASE}/agents`);
  return response.json();
}

export function getDebateStreamUrl(id: string): string {
  return `${API_BASE}/debates/${id}/stream`;
}
