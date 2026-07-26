export interface DocumentRecord {
  doc_id: string;
  file_name: string;
  size_bytes: number;
  pages: number;
  chunks: number;
  chars: number;
  ingested_at: string;
}

export interface IndexStats {
  documents: number;
  pages: number;
  chunks: number;
  chars: number;
  embed_model: string;
  llm_model: string;
  chunk_size: number;
  chunk_overlap: number;
}

export interface DocumentsResponse {
  documents: DocumentRecord[];
  stats: IndexStats;
}

export interface UploadResult {
  status: "indexed" | "duplicate" | "error";
  file_name?: string;
  doc_id?: string;
  detail?: string;
  pages?: number;
  chunks?: number;
  chars?: number;
}

export interface UploadResponse {
  results: UploadResult[];
  stats: IndexStats;
}

export interface Source {
  position: number;
  doc_id: string;
  file_name: string;
  page: number;
  score: number;
  text: string;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  elapsed_seconds: number;
}

export interface SearchResponse {
  mode: string;
  hits: Source[];
  elapsed_seconds: number;
}

export interface RustStatus {
  binary: string;
  binary_ready: boolean;
  model: string;
  model_ready: boolean;
}

export interface RustResponse {
  file_name: string;
  answer: string;
  chars_extracted: number;
  chars_sent: number;
  truncated: boolean;
  elapsed_seconds: number;
}

export interface AgentInfo {
  key: string;
  label: string;
  binary: string;
  model: string;
  default_model: string;
  installed: boolean;
}

export interface AgentPreferences {
  active: string;
  models: Record<string, string>;
  timeout: number;
}

export interface AgentConfigResponse {
  agents: AgentInfo[];
  preferences: AgentPreferences;
}

export interface AgentAnswer {
  agent: string;
  label: string;
  model: string;
  answer: string;
  elapsed_seconds: number;
  exit_code: number;
}

export interface Mark {
  page: number;
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
  note: string;
  kind: "highlight" | "note";
}

export interface AnnotationSaved {
  name: string;
  applied: number;
  size_bytes: number;
  url: string;
}

export interface AnnotationFile {
  name: string;
  size_bytes: number;
}

export interface Health {
  ok: boolean;
  ollama: { host: string; reachable: boolean };
  index: IndexStats;
}
