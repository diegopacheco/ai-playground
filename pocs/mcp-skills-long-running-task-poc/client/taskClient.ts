import { CreateTaskResultV2Schema, GetTaskResultV2Schema, type CreateTaskResultV2, type GetTaskResultV2 } from "@modelcontextprotocol/ext-tasks/core/v2";
import type { RawDispatcher } from "./rawRpc.ts";

export type TaskSnapshot = GetTaskResultV2;
export type Wait = (ms: number) => Promise<void>;

const DEFAULT_POLL_MS = 500;

export class TaskClient {
  constructor(private readonly rpc: RawDispatcher, private readonly wait: Wait) {}

  async callTool(name: string, args: Readonly<Record<string, unknown>>): Promise<CreateTaskResultV2> {
    return CreateTaskResultV2Schema.parse(await this.rpc.request("tools/call", { name, arguments: args }));
  }

  async get(taskId: string): Promise<TaskSnapshot> {
    return GetTaskResultV2Schema.parse(await this.rpc.request("tasks/get", { taskId }));
  }

  async cancel(taskId: string): Promise<void> {
    await this.rpc.request("tasks/cancel", { taskId });
  }

  async waitUntilDone(handle: CreateTaskResultV2, onUpdate: (task: TaskSnapshot) => void): Promise<TaskSnapshot> {
    const pollMs = handle.pollIntervalMs ?? DEFAULT_POLL_MS;
    for (;;) {
      await this.wait(pollMs);
      const task = await this.get(handle.taskId);
      onUpdate(task);
      if (task.status !== "working" && task.status !== "input_required") return task;
    }
  }
}
