import { proxyActivities } from '@temporalio/workflow';
import type * as activities from './activities.ts';
import type { AgentPipelineInput, AgentPipelineResult } from './shared.ts';

const { researchTopic, writeDraft, critiqueDraft, editDraft } = proxyActivities<typeof activities>({
  startToCloseTimeout: '2 minutes',
  retry: {
    initialInterval: '2 seconds',
    maximumAttempts: 3,
  },
});

export async function multiAgentPipeline(input: AgentPipelineInput): Promise<AgentPipelineResult> {
  const research = await researchTopic(input.topic);
  const draft = await writeDraft(input.topic, research);
  const critique = await critiqueDraft(draft);
  const finalArticle = await editDraft(draft, critique);

  return {
    topic: input.topic,
    research,
    draft,
    critique,
    finalArticle,
  };
}
