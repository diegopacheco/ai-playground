import type { Decision, PolicyEvent } from "./types";

const LINE = /^\[(\d+(?:\.\d+)?)\] \[sandbox\] \[OCSF\s*\] \[ocsf\] (\S+) \[(\w+)\] (ALLOWED|DENIED) (.*)$/;
const POLICY = /\[policy:([^\s\]]+)/;
const ENGINE = /engine:([^\s\]]+)/;
const REASON = /\[reason:(.*)\]$/;

export function parsePolicyEvents(text: string): PolicyEvent[] {
  return text
    .split("\n")
    .map(parseLine)
    .filter((event): event is PolicyEvent => event !== null);
}

function parseLine(line: string): PolicyEvent | null {
  const match = LINE.exec(line.trim());
  if (!match) {
    return null;
  }
  const [, seconds, kind, severity, decision, rest] = match;
  const bracket = rest.indexOf(" [");
  return {
    timeMs: Math.round(Number(seconds) * 1000),
    kind,
    severity,
    decision: decision as Decision,
    target: bracket === -1 ? rest : rest.slice(0, bracket),
    policy: POLICY.exec(rest)?.[1] ?? "-",
    engine: ENGINE.exec(rest)?.[1] ?? "-",
    reason: REASON.exec(rest)?.[1] ?? "",
  };
}
