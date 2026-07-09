import { Client, Connection } from '@temporalio/client';
import { TASK_QUEUE, type AgentPipelineResult } from './shared.ts';

const address = process.env.TEMPORAL_ADDRESS ?? 'localhost:7233';

async function run() {
  const topic = process.argv.slice(2).join(' ') || 'Durable execution with Temporal';

  const connection = await Connection.connect({ address });
  const client = new Client({ connection });

  const handle = await client.workflow.start('multiAgentPipeline', {
    taskQueue: TASK_QUEUE,
    workflowId: `agent-pipeline-${Date.now()}`,
    args: [{ topic }],
  });

  console.log(`Started workflow ${handle.workflowId}`);
  console.log('Waiting for the agents to finish (watch http://localhost:8233)...\n');

  const result = (await handle.result()) as AgentPipelineResult;

  console.log('===== RESEARCH (research agent) =====\n' + result.research);
  console.log('\n===== DRAFT (writing agent) =====\n' + result.draft);
  console.log('\n===== CRITIQUE (critique agent) =====\n' + result.critique);
  console.log('\n===== FINAL ARTICLE (editor agent) =====\n' + result.finalArticle);

  await connection.close();
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
