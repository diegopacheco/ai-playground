export interface PrInfo {
  url: string;
  owner: string;
  repo: string;
  pr_number: number;
  title: string;
  branch: string;
  total_files: number;
  clone_path: string;
  agent_name: string;
  agent_model: string;
}

export type ActionType =
  | "CompileFix"
  | "TestFix"
  | "TestAdd"
  | "CommentReply"
  | "MergeConflictFix";

export interface AgentAction {
  id: number;
  timestamp: string;
  action_type: ActionType;
  description: string;
  files_changed: string[];
  llm_agent: string;
  llm_model: string;
  commit_sha: string | null;
}

export interface Counters {
  compilation_fixes: number;
  test_fixes: number;
  tests_added: number;
  comments_answered: number;
  total_cycles: number;
}

export interface TestClassification {
  unit: number;
  integration: number;
  e2e: number;
  other: number;
}

export interface CommentReply {
  author: string;
  body: string;
  timestamp: string;
  is_agent: boolean;
}

export interface CommentThread {
  id: number;
  github_comment_id: number;
  author: string;
  body: string;
  file_path: string | null;
  line: number | null;
  timestamp: string;
  replies: CommentReply[];
}

export interface FileEntry {
  path: string;
  name: string;
  is_dir: boolean;
  children: FileEntry[];
}

export interface AgentLog {
  id: number;
  timestamp: string;
  action_type: ActionType;
  llm_agent: string;
  llm_model: string;
  prompt: string;
  response: string;
  result: string;
  commit_sha: string | null;
}

export interface StatusResponse {
  pr_info: PrInfo;
  counters: Counters;
  test_classification: TestClassification;
}

export interface HealthResponse {
  uptime_seconds: number;
  total_cycles: number;
  last_check: string | null;
}

export interface ConversationEntry {
  type: "description" | "issue" | "review";
  id: number;
  author: string;
  body: string;
  created_at: string;
  file_path: string | null;
  line: number | null;
}
