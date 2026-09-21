export interface AskResult {
  question: string;
  answer: string;
  cached: boolean;
  similarity: number | null;
  matchedQuestion: string | null;
  latencyMs: number;
}

export interface CacheEntry {
  key: string;
  question: string;
  answer: string;
  hits: number;
  createdAt: number;
}

export interface Stats {
  hits: number;
  misses: number;
  entries: number;
  threshold: number;
  claudeModel: string;
  embedModel: string;
}
