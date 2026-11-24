import type { Clock, IdGenerator } from "../clock.ts";
import type { Logger } from "../logger.ts";
import { isTerminal, type Job, type Task, type TaskId } from "./task.ts";

type Entry = { readonly task: Task; readonly controller: AbortController; readonly done: Promise<void> };

export type TaskStoreDeps = {
  readonly clock: Clock;
  readonly ids: IdGenerator;
  readonly logger: Logger;
  readonly ttlMs: number;
};

export class TaskStore {
  readonly #entries = new Map<TaskId, Entry>();

  constructor(private readonly deps: TaskStoreDeps) {}

  start(job: Job): Task {
    const now = this.deps.clock.now();
    const task: Task = { id: this.deps.ids(), status: "working", message: "queued", createdAt: now, updatedAt: now };
    const controller = new AbortController();
    const done = this.#run(task.id, job, controller.signal);
    this.#entries.set(task.id, { task, controller, done });
    this.deps.logger.info("task started", { taskId: task.id });
    return task;
  }

  get(id: TaskId): Task | undefined {
    const entry = this.#entries.get(id);
    if (!entry) return undefined;
    if (this.#expired(entry.task)) {
      entry.controller.abort();
      this.#entries.delete(id);
      return undefined;
    }
    return entry.task;
  }

  cancel(id: TaskId): Task | undefined {
    const task = this.get(id);
    if (!task || isTerminal(task)) return task;
    this.#entries.get(id)?.controller.abort();
    return this.#transition(id, (t) => ({ ...t, status: "cancelled", message: "cancelled by client", updatedAt: this.deps.clock.now() }));
  }

  async settled(id: TaskId): Promise<Task | undefined> {
    await this.#entries.get(id)?.done;
    return this.get(id);
  }

  async #run(id: TaskId, job: Job, signal: AbortSignal): Promise<void> {
    await Promise.resolve();
    if (signal.aborted) return;
    const update = (fn: (t: Task) => Task) => this.#transition(id, (t) => (t.status === "working" ? fn(t) : t));
    try {
      const output = await job({ signal, report: (message) => update((t) => ({ ...t, message, updatedAt: this.deps.clock.now() })) });
      update((t) => ({ ...t, status: "completed", message: "done", output, updatedAt: this.deps.clock.now() }));
      this.deps.logger.info("task completed", { taskId: id });
    } catch (err) {
      if (signal.aborted) return;
      const error = err instanceof Error ? err.message : String(err);
      update((t) => ({ ...t, status: "failed", message: "failed", error, updatedAt: this.deps.clock.now() }));
      this.deps.logger.error("task failed", { taskId: id, error });
    }
  }

  #transition(id: TaskId, fn: (t: Task) => Task): Task | undefined {
    const entry = this.#entries.get(id);
    if (!entry) return undefined;
    const task = fn(entry.task);
    this.#entries.set(id, { ...entry, task });
    return task;
  }

  #expired(task: Task): boolean {
    return this.deps.clock.now().getTime() - task.createdAt.getTime() > this.deps.ttlMs;
  }
}
