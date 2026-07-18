import type { ConnectionKind } from "@lib/types";
import type { EngineDescriptor } from "./types";
import { mysql } from "./mysql";
import { postgres } from "./postgres";
import { cassandra } from "./cassandra";
import { redis } from "./redis";
import { etcd } from "./etcd";
import { kafka } from "./kafka";
import { elasticsearch } from "./elasticsearch";

const descriptors: Record<ConnectionKind, EngineDescriptor> = {
  mysql,
  postgres,
  cassandra,
  redis,
  etcd,
  kafka,
  elasticsearch
};

export function engineFor(kind: ConnectionKind): EngineDescriptor {
  return descriptors[kind];
}

export const allEngines = Object.values(descriptors);
