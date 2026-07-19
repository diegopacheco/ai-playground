import type { EngineDescriptor } from "./types";
import { flattenNames } from "./types";

const READ_COMMANDS = [
  "GET", "MGET", "STRLEN", "EXISTS", "TYPE", "TTL", "PTTL",
  "HGETALL", "HGET", "HKEYS", "HVALS", "HLEN",
  "LRANGE", "LLEN", "LINDEX",
  "SMEMBERS", "SCARD", "SISMEMBER",
  "ZRANGE", "ZCARD", "ZSCORE", "ZRANGEBYSCORE",
  "XRANGE", "XLEN", "SCAN", "DBSIZE", "INFO"
];

export const redis: EngineDescriptor = {
  kind: "redis",
  label: "Redis",
  language: "command",
  placeholder: "HGETALL session:abc123",
  keywords: READ_COMMANDS,
  schemaLabel: "keys",
  supportsCount: false,
  emptySchemaLabel: "no keys in this database",
  sampleStatement: (schema) => {
    const key = schema[0];
    if (!key) {
      return "DBSIZE";
    }
    return key.kind === "hash" ? `HGETALL ${key.name}` : `GET ${key.name}`;
  },
  completionsFor: (schema) => [...READ_COMMANDS, ...flattenNames(schema, 1)]
};
