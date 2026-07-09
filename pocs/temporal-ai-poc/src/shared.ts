export const TASK_QUEUE = 'agent-pipeline';

export interface AgentPipelineInput {
  topic: string;
}

export interface AgentPipelineResult {
  topic: string;
  research: string;
  draft: string;
  critique: string;
  finalArticle: string;
}
