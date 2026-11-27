import { list, num, obj, oneOf, str } from "../../parse";

export const ACTOR_STATES = [
  "RESUMING",
  "RUNNING",
  "SUSPENDING",
  "SUSPENDED",
  "CRASHED",
  "DELETING",
  "REVERTING",
  "UNKNOWN",
] as const;
export type ActorState = (typeof ACTOR_STATES)[number];

export type Actor = {
  readonly name: string;
  readonly template: string;
  readonly state: ActorState;
  readonly workerPod: string;
  readonly node: string;
  readonly version: number;
  readonly snapshots: number;
  readonly createdAt: string;
};

export type Worker = {
  readonly name: string;
  readonly pool: string;
  readonly pod: string;
  readonly node: string;
  readonly state: string;
  readonly actors: number;
  readonly capacity: number;
};

export type Note = { readonly id: number; readonly text: string; readonly at: string };

export type AgentState = {
  readonly bootId: string;
  readonly startedAt: string;
  readonly hostname: string;
  readonly requests: number;
  readonly memoryNotes: readonly Note[];
  readonly journalNotes: readonly Note[];
};

export function parseActor(raw: unknown): Actor {
  const o = obj(raw, "actor");
  return {
    name: str(o, "name", "actor"),
    template: str(o, "template", "actor"),
    state: oneOf(o, "state", ACTOR_STATES, "actor state"),
    workerPod: str(o, "workerPod", "actor"),
    node: str(o, "node", "actor"),
    version: num(o, "version", "actor"),
    snapshots: num(o, "snapshots", "actor"),
    createdAt: str(o, "createdAt", "actor"),
  };
}

export function parseActors(raw: unknown): readonly Actor[] {
  return list(raw ?? [], "actors", parseActor);
}

function parseWorker(raw: unknown): Worker {
  const o = obj(raw, "worker");
  return {
    name: str(o, "name", "worker"),
    pool: str(o, "pool", "worker"),
    pod: str(o, "pod", "worker"),
    node: str(o, "node", "worker"),
    state: str(o, "state", "worker"),
    actors: num(o, "actors", "worker"),
    capacity: num(o, "capacity", "worker"),
  };
}

export function parseWorkers(raw: unknown): readonly Worker[] {
  return list(raw ?? [], "workers", parseWorker);
}

export function parseNote(raw: unknown): Note {
  const o = obj(raw, "note");
  return { id: num(o, "id", "note"), text: str(o, "text", "note"), at: str(o, "at", "note") };
}

export function parseAgentState(raw: unknown): AgentState {
  const o = obj(raw, "agent state");
  return {
    bootId: str(o, "bootId", "agent state"),
    startedAt: str(o, "startedAt", "agent state"),
    hostname: str(o, "hostname", "agent state"),
    requests: num(o, "requests", "agent state"),
    memoryNotes: list(o.memoryNotes ?? [], "agent state", parseNote),
    journalNotes: list(o.journalNotes ?? [], "agent state", parseNote),
  };
}

export function parseDeleted(raw: unknown): string {
  return str(obj(raw, "delete"), "deleted", "delete");
}
