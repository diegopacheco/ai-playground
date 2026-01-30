export interface DimensionScores {
  quality: number;
  stack_definitions: number;
  clear_goals: number;
  non_obvious_decisions: number;
  security_operations: number;
  overall_effectiveness: number;
}

export interface ModelResult {
  model: string;
  score: number | null;
  recommendations: string;
}

export interface AnalyzeResponse {
  scores: DimensionScores;
  model_results: ModelResult[];
}

export type ProgressEvent =
  | { type: 'start'; total_steps: number; message: string }
  | { type: 'scores'; scores: DimensionScores; step: number; message: string }
  | { type: 'agent_start'; agent: string; step: number; message: string }
  | { type: 'agent_done'; agent: string; result: ModelResult; step: number; message: string }
  | { type: 'complete'; scores: DimensionScores; model_results: ModelResult[]; message: string };

export interface ProgressState {
  currentStep: number;
  totalSteps: number;
  message: string;
  isComplete: boolean;
}
