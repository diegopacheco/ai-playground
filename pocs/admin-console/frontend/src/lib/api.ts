import {
  ApiError,
  type AiCli,
  type AiSuggestion,
  type AuditEntry,
  type DiscoveredContainer,
  type Project,
  type QueryResult,
  type SavedQuery,
  type SchemaNode,
  type Session
} from "./types";

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    credentials: "include",
    headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
    ...init
  });
  const text = await response.text();
  const body = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const message = body?.error ?? `request failed with ${response.status}`;
    throw new ApiError(message, response.status, Boolean(body?.readOnlyViolation));
  }
  return body as T;
}

export const api = {
  login: (username: string, password: string) =>
    request<Session>("/api/auth/login", { method: "POST", body: JSON.stringify({ username, password }) }),

  logout: () => request<unknown>("/api/auth/logout", { method: "POST" }),

  session: () => request<Session>("/api/auth/session"),

  changeOwnPassword: (password: string) =>
    request<unknown>("/api/auth/password", { method: "POST", body: JSON.stringify({ password }) }),

  projects: () => request<Project[]>("/api/projects"),

  createProject: (name: string) =>
    request<{ id: number }>("/api/projects", { method: "POST", body: JSON.stringify({ name }) }),

  deleteProject: (id: number) => request<unknown>(`/api/projects/${id}`, { method: "DELETE" }),

  addConnection: (projectId: number, connection: Record<string, unknown>) =>
    request<unknown>(`/api/projects/${projectId}/connections`, {
      method: "POST",
      body: JSON.stringify(connection)
    }),

  deleteConnection: (projectId: number, connectionId: number) =>
    request<unknown>(`/api/projects/${projectId}/connections/${connectionId}`, { method: "DELETE" }),

  schema: (connectionId: number) => request<SchemaNode[]>(`/api/connections/${connectionId}/schema`),

  ping: (connectionId: number) =>
    request<{ healthy: boolean; error?: string }>(`/api/connections/${connectionId}/ping`),

  query: (
    connectionId: number,
    statement: string,
    options: { pageSize?: number; cursor?: string | null; pageNumber?: number; queryId?: string } = {}
  ) =>
    request<QueryResult>(`/api/connections/${connectionId}/query`, {
      method: "POST",
      body: JSON.stringify({ statement, ...options })
    }),

  count: (connectionId: number, statement: string) =>
    request<{ total: number }>(
      `/api/connections/${connectionId}/count?statement=${encodeURIComponent(statement)}`
    ),

  history: (connectionId?: number, limit = 20) =>
    request<string[]>(
      `/api/history?limit=${limit}${connectionId ? `&connection=${connectionId}` : ""}`
    ),

  audit: (params: Record<string, string | number | boolean | undefined> = {}) => {
    const query = Object.entries(params)
      .filter(([, value]) => value !== undefined && value !== "")
      .map(([key, value]) => `${key}=${encodeURIComponent(String(value))}`)
      .join("&");
    return request<{ page: number; size: number; entries: AuditEntry[] }>(`/api/audit${query ? `?${query}` : ""}`);
  },

  discover: () =>
    request<{ runtime: string | null; available: boolean; reason?: string; containers: DiscoveredContainer[] }>(
      "/api/discovery"
    ),

  importDiscovered: (projectName: string, containerIds: string[]) =>
    request<{ projectId: number; projectName: string; imported: string[] }>("/api/discovery/import", {
      method: "POST",
      body: JSON.stringify({ projectName, containerIds })
    }),

  savedQueries: (projectId: number) => request<SavedQuery[]>(`/api/projects/${projectId}/saved`),

  saveQuery: (projectId: number, body: Record<string, unknown>) =>
    request<SavedQuery>(`/api/projects/${projectId}/saved`, { method: "POST", body: JSON.stringify(body) }),

  deleteSavedQuery: (projectId: number, id: number) =>
    request<unknown>(`/api/projects/${projectId}/saved/${id}`, { method: "DELETE" }),

  aiClis: () => request<AiCli[]>("/api/ai/clis"),

  aiSettings: () =>
    request<{ cli: string; label: string; model: string | null; usingDefault: boolean; installed: boolean }>(
      "/api/ai/settings"
    ),

  saveAiSettings: (cli: string, model: string) =>
    request<unknown>("/api/ai/settings", { method: "PUT", body: JSON.stringify({ cli, model }) }),

  saveGlobalAiSettings: (cli: string, model: string, enabled: boolean) =>
    request<unknown>("/api/ai/settings/global", {
      method: "PUT",
      body: JSON.stringify({ cli, model, enabled })
    }),

  aiQuery: (connectionId: number, prompt: string) =>
    request<AiSuggestion>(`/api/ai/connections/${connectionId}/query`, {
      method: "POST",
      body: JSON.stringify({ prompt })
    }),

  users: () => request<Record<string, unknown>[]>("/api/users"),

  createUser: (username: string, password: string, role: string) =>
    request<unknown>("/api/users", { method: "POST", body: JSON.stringify({ username, password, role }) }),

  deleteUser: (id: number) => request<unknown>(`/api/users/${id}`, { method: "DELETE" })
};
