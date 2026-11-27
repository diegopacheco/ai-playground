import type { ActorState, AgentState } from "./types";

export type Action = "suspend" | "resume" | "delete";

export function allowedActions(state: ActorState): readonly Action[] {
  switch (state) {
    case "RUNNING":
      return ["suspend", "delete"];
    case "SUSPENDED":
    case "CRASHED":
      return ["resume", "delete"];
    case "RESUMING":
    case "SUSPENDING":
    case "DELETING":
    case "REVERTING":
      return [];
    case "UNKNOWN":
      return ["delete"];
    default: {
      const unreachable: never = state;
      return unreachable;
    }
  }
}

export type MemoryVerdict = "empty" | "restored" | "cold-boot";

export function memoryVerdict(state: AgentState): MemoryVerdict {
  if (state.journalNotes.length === 0) return "empty";
  return state.memoryNotes.length === state.journalNotes.length ? "restored" : "cold-boot";
}

export function isTransient(state: ActorState): boolean {
  return state === "RESUMING" || state === "SUSPENDING" || state === "DELETING" || state === "REVERTING";
}

const DNS_LABEL = /^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$/;

export function nameError(name: string): string | undefined {
  if (name === "") return "Name is required";
  return DNS_LABEL.test(name) ? undefined : "Use lower case letters, digits and dashes";
}
