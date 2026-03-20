import { useState, useEffect, useRef } from "react";
import type { CommitResult } from "../types";

interface SSECommit {
  sha: string;
  message: string;
  repo: string;
}

interface SSEResult {
  index: number;
  sha: string;
  classification: string;
  summary: string;
  score: number;
  fallback: boolean;
}

interface SSEScores {
  total_score: number;
  avg_score: number;
  deep: number;
  decent: number;
  shallow: number;
}

export type SSEPhase =
  | "idle"
  | "started"
  | "commits_fetched"
  | "llm_thinking"
  | "analyzed"
  | "complete"
  | "error";

interface SSEState {
  phase: SSEPhase;
  commits: SSECommit[];
  results: SSEResult[];
  scores: SSEScores | null;
  error: string | null;
}

export function useAnalysisSSE(analysisId: string | null): SSEState {
  const [state, setState] = useState<SSEState>({
    phase: "idle",
    commits: [],
    results: [],
    scores: null,
    error: null,
  });
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!analysisId) {
      setState({
        phase: "idle",
        commits: [],
        results: [],
        scores: null,
        error: null,
      });
      return;
    }

    const es = new EventSource(
      `http://localhost:3000/api/analyze/${analysisId}/stream`
    );
    eventSourceRef.current = es;

    es.addEventListener("analysis_start", () => {
      setState((prev) => ({ ...prev, phase: "started" }));
    });

    es.addEventListener("commits_fetched", (e) => {
      const commits = JSON.parse(e.data);
      setState((prev) => ({ ...prev, phase: "commits_fetched", commits }));
    });

    es.addEventListener("llm_thinking", () => {
      setState((prev) => ({ ...prev, phase: "llm_thinking" }));
    });

    es.addEventListener("all_commits_analyzed", (e) => {
      const results = JSON.parse(e.data);
      setState((prev) => ({ ...prev, phase: "analyzed", results }));
    });

    es.addEventListener("analysis_complete", (e) => {
      const scores = JSON.parse(e.data);
      setState((prev) => ({ ...prev, phase: "complete", scores }));
      es.close();
    });

    es.addEventListener("error", (e) => {
      const data = (e as MessageEvent).data;
      let message = "Unknown error";
      try {
        message = JSON.parse(data).message;
      } catch {
        /* keep default */
      }
      setState((prev) => ({ ...prev, phase: "error", error: message }));
      es.close();
    });

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  }, [analysisId]);

  return state;
}
