import { NativeConnection, Worker } from '@temporalio/worker';
import * as activities from './activities.ts';
import { TASK_QUEUE } from './shared.ts';

const address = process.env.TEMPORAL_ADDRESS ?? 'localhost:7233';

async function run() {
  const connection = await NativeConnection.connect({ address });
  const worker = await Worker.create({
    connection,
    namespace: 'default',
    taskQueue: TASK_QUEUE,
    workflowsPath: new URL('./workflows.ts', import.meta.url).pathname,
    activities,
  });

  console.log(`Worker connected to ${address}, polling task queue "${TASK_QUEUE}"`);
  await worker.run();
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
