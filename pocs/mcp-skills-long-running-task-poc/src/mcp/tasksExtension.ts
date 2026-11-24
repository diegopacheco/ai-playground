import type { McpServer, ServerContext } from "@modelcontextprotocol/server";
import { TASKS_EXTENSION_ID_V2, hasTaskClientCapabilityV2 } from "@modelcontextprotocol/ext-tasks/core/v2";
import * as z from "zod";
import type { TaskStore } from "../tasks/store.ts";
import type { Task } from "../tasks/task.ts";

export type TaskTiming = { readonly ttlMs: number; readonly pollIntervalMs: number };

const TaskIdParams = z.object({ taskId: z.string().min(1) });

export class TaskNotFoundError extends Error {
  constructor(readonly taskId: string) {
    super(`task ${taskId} not found`);
  }
}

function base(task: Task, timing: TaskTiming) {
  return {
    taskId: task.id,
    statusMessage: task.message,
    createdAt: task.createdAt.toISOString(),
    lastUpdatedAt: task.updatedAt.toISOString(),
    ttlMs: timing.ttlMs,
    pollIntervalMs: timing.pollIntervalMs,
  };
}

export function toWireTask(task: Task, timing: TaskTiming) {
  switch (task.status) {
    case "working":
      return { ...base(task, timing), status: task.status };
    case "completed":
      return { ...base(task, timing), status: task.status, result: { content: [{ type: "text", text: task.output.text }], isError: false } };
    case "failed":
      return { ...base(task, timing), status: task.status, error: { code: -32603, message: task.error } };
    case "cancelled":
      return { ...base(task, timing), status: task.status };
    default: {
      const unreachable: never = task;
      return unreachable;
    }
  }
}

export function createTaskResult(task: Task, timing: TaskTiming) {
  return { ...base(task, timing), status: task.status, resultType: "task" as const };
}

export function clientAcceptsTasks(ctx: ServerContext): boolean {
  return hasTaskClientCapabilityV2({ _meta: ctx.mcpReq.envelope ?? {} });
}

function found(store: TaskStore, taskId: string, lookup: (id: string) => Task | undefined): Task {
  const task = lookup.call(store, taskId);
  if (!task) throw new TaskNotFoundError(taskId);
  return task;
}

export function attachTasks(mcp: McpServer, store: TaskStore, timing: TaskTiming): void {
  mcp.server.registerCapabilities({ extensions: { [TASKS_EXTENSION_ID_V2]: {} } });
  mcp.server.setRequestHandler("tasks/get", { params: TaskIdParams }, async ({ taskId }) => toWireTask(found(store, taskId, store.get), timing));
  mcp.server.setRequestHandler("tasks/cancel", { params: TaskIdParams }, async ({ taskId }) => {
    found(store, taskId, store.cancel);
    return {};
  });
}
