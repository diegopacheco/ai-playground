export type Route = { kind: "playbook"; id: string; matched: string[] } | { kind: "figure-it-out"; matched: string[] };

type Rule = { id: string; keywords: readonly string[] };

const BROAD: readonly string[] = ["migrate", "rewrite", "entire", "across the codebase", "overnight", "multi-day"];

const RULES: readonly Rule[] = [
  { id: "bug-fix", keywords: ["bug", "broken", "crash", "error", "fails", "wrong", "twice", "double"] },
  { id: "perf-issue", keywords: ["slow", "perf", "latency", "faster", "memory", "cpu"] },
  { id: "investigation", keywords: ["why", "how does", "explain", "investigate", "understand", "where is"] },
  { id: "refactoring", keywords: ["refactor", "clean up", "rename", "extract", "simplify"] },
  { id: "prototype", keywords: ["prototype", "spike", "try out", "compare"] },
  { id: "hillclimb", keywords: ["hillclimb", "eval score", "raise the score", "pass rate"] },
  { id: "visual-parity", keywords: ["pixel", "figma", "match the design", "visual"] },
  { id: "session-pickup", keywords: ["pick up", "resume", "where did i leave"] },
  { id: "pause-safely", keywords: ["pause", "stop for today", "park this"] },
  { id: "feature", keywords: ["add", "build", "implement", "support", "new"] },
];

const escape = (w: string): string => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const hits = (task: string, words: readonly string[]): string[] =>
  words.filter((w) => new RegExp(`\\b${escape(w)}\\b`).test(task));

export function route(task: string): Route {
  const text = task.toLowerCase();
  const broad = hits(text, BROAD);
  if (broad.length > 0) return { kind: "figure-it-out", matched: broad };
  let best: Route = { kind: "figure-it-out", matched: [] };
  let bestCount = 0;
  for (const rule of RULES) {
    const matched = hits(text, rule.keywords);
    if (matched.length > bestCount) {
      best = { kind: "playbook", id: rule.id, matched };
      bestCount = matched.length;
    }
  }
  return best;
}

export type Trigger = { skill: string; reason: string };

const TRIGGERS: readonly { skill: string; reason: string; keywords: readonly string[] }[] = [
  { skill: "how", reason: "nontrivial change: explain how the code works first", keywords: ["fix", "add", "refactor", "change", "bug", "slow"] },
  { skill: "architect", reason: "code crosses a function boundary", keywords: ["api", "module", "service", "endpoint", "component"] },
  { skill: "interrogate", reason: "contested design", keywords: ["should we", "or", "debate", "tradeoff"] },
  { skill: "benchmark-checklist", reason: "a number will be measured", keywords: ["slow", "faster", "latency", "ms", "benchmark"] },
  { skill: "unslop", reason: "prose goes out, the reply included", keywords: [] },
];

export function triggers(task: string): Trigger[] {
  const text = task.toLowerCase();
  return TRIGGERS.filter((t) => t.keywords.length === 0 || hits(text, t.keywords).length > 0).map(({ skill, reason }) => ({ skill, reason }));
}
