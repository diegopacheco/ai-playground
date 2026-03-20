export interface CommitResult {
  id: string;
  analysis_id: string;
  repo_name: string;
  commit_sha: string;
  commit_message: string;
  commit_date: string;
  classification: "DEEP" | "DECENT" | "SHALLOW";
  summary: string;
  score: number;
  fallback: boolean;
  raw_llm_output: string | null;
}

export interface Analysis {
  id: string;
  github_user: string;
  created_at: string;
  total_score: number;
  avg_score: number;
  status: "running" | "completed" | "failed";
  commits: CommitResult[];
}

export interface WeeklyScore {
  id: string;
  github_user: string;
  week_start: string;
  week_end: string;
  total_score: number;
  avg_score: number;
  num_commits: number;
  deep_count: number;
  decent_count: number;
  shallow_count: number;
}

export interface LeaderboardEntry {
  rank: number;
  github_user: string;
  avg_score: number;
  total_score: number;
  num_commits: number;
  deep_count: number;
  decent_count: number;
  shallow_count: number;
}

export interface AnalyzeResponse {
  id: string;
}

export interface TrackedUser {
  github_user: string;
  added_at: string;
}
