import type { SearchRequest, SearchResponse, AgentInfo } from '../types';

const API_BASE = 'http://localhost:3000/api';

export async function searchFlights(req: SearchRequest): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return response.json();
}

export async function getAgents(): Promise<AgentInfo[]> {
  const response = await fetch(`${API_BASE}/agents`);
  return response.json();
}
