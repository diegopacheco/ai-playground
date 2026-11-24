export type TaskId = string;

export type TaskOutput = { readonly text: string };

type TaskBase = {
  readonly id: TaskId;
  readonly message: string;
  readonly createdAt: Date;
  readonly updatedAt: Date;
};

export type Task =
  | (TaskBase & { readonly status: "working" })
  | (TaskBase & { readonly status: "completed"; readonly output: TaskOutput })
  | (TaskBase & { readonly status: "failed"; readonly error: string })
  | (TaskBase & { readonly status: "cancelled" });

export type TaskStatus = Task["status"];

export type JobContext = {
  readonly signal: AbortSignal;
  readonly report: (message: string) => void;
};

export type Job = (ctx: JobContext) => Promise<TaskOutput>;

export function isTerminal(task: Task): boolean {
  return task.status !== "working";
}
