export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export interface Stored {
  format: string;
  readable: boolean;
  value: Json;
}

export interface WorkflowRow {
  id: string;
  name: string;
  status: string;
  created_at: number;
  updated_at: number;
  completed_at: number | null;
  executor: string;
  app_version: string;
  recovery_attempts: number;
  queue: string;
  steps: number;
}

export interface StatusCount {
  status: string;
  count: number;
}

export interface WorkflowList {
  counts: StatusCount[];
  workflows: WorkflowRow[];
}

export interface StepRow {
  id: number;
  name: string;
  output: Stored;
  error: Stored;
  child_workflow: string;
  started_at: number | null;
  completed_at: number | null;
}

export interface EventRow {
  function_id: number | null;
  key: string;
  value: Stored;
}

export interface WorkflowDetail {
  workflow: WorkflowRow;
  serialization: string;
  input: Stored;
  output: Stored;
  error: Stored;
  steps: StepRow[];
  events: EventRow[];
  history: EventRow[];
}

export interface WorkflowFilter {
  status: string;
  search: string;
}
