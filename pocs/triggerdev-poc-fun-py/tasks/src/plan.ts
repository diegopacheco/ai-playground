export const MAX_TASKS = 4;

export function parsePlan(output: string): string[] {
  const start = output.indexOf("[");
  const end = output.lastIndexOf("]");
  if (start < 0 || end < start) throw new Error(`agent returned no JSON task list: ${output.slice(0, 200)}`);
  const value: unknown = JSON.parse(output.slice(start, end + 1));
  if (!Array.isArray(value)) throw new Error("agent plan is not a list");
  const tasks = value.filter((item): item is string => typeof item === "string").map(item => item.trim()).filter(Boolean);
  if (tasks.length === 0) throw new Error("agent plan has no tasks");
  return tasks.slice(0, MAX_TASKS);
}
