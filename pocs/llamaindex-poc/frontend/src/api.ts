import type {
  AgentAnswer,
  AgentConfigResponse,
  AnnotationFile,
  AnnotationSaved,
  ChatResponse,
  DocumentsResponse,
  Health,
  Mark,
  RustResponse,
  RustStatus,
  SearchResponse,
  UploadResponse,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

function post<T>(path: string, payload: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export const api = {
  health: (): Promise<Health> => request<Health>("/api/health"),

  documents: (): Promise<DocumentsResponse> => request<DocumentsResponse>("/api/documents"),

  upload: (files: File[]): Promise<UploadResponse> => {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    return request<UploadResponse>("/api/documents", { method: "POST", body: form });
  },

  deleteDocument: (docId: string): Promise<unknown> =>
    request(`/api/documents/${docId}`, { method: "DELETE" }),

  chat: (payload: {
    question: string;
    history: { role: string; content: string }[];
    top_k: number;
    doc_ids: string[];
  }): Promise<ChatResponse> => post<ChatResponse>("/api/chat", payload),

  search: (payload: {
    query: string;
    mode: string;
    top_k: number;
    doc_ids: string[];
  }): Promise<SearchResponse> => post<SearchResponse>("/api/search", payload),

  rustStatus: (): Promise<RustStatus> => request<RustStatus>("/api/rust/status"),

  rustAsk: (payload: { doc_id: string; question: string }): Promise<RustResponse> =>
    post<RustResponse>("/api/rust/ask", payload),

  agents: (): Promise<AgentConfigResponse> => request<AgentConfigResponse>("/api/agents"),

  saveAgents: (payload: {
    active?: string;
    models?: Record<string, string>;
    timeout?: number;
  }): Promise<AgentConfigResponse> =>
    request<AgentConfigResponse>("/api/agents", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    }),

  askAgent: (payload: { prompt: string; agent?: string; context?: string }): Promise<AgentAnswer> =>
    post<AgentAnswer>("/api/agents/ask", payload),

  annotations: (): Promise<{ files: AnnotationFile[] }> =>
    request<{ files: AnnotationFile[] }>("/api/annotations"),

  saveAnnotations: (payload: { doc_id: string; marks: Mark[] }): Promise<AnnotationSaved> =>
    post<AnnotationSaved>("/api/annotations", payload),
};
