import type { EngineDescriptor } from "./types";
import { flattenNames } from "./types";

const CQL_KEYWORDS = [
  "SELECT", "FROM", "WHERE", "AND", "IN", "ORDER BY", "LIMIT", "ALLOW FILTERING",
  "TOKEN", "COUNT", "DISTINCT", "ASC", "DESC"
];

export const cassandra: EngineDescriptor = {
  kind: "cassandra",
  label: "Cassandra",
  language: "sql",
  placeholder: "SELECT * FROM events_by_customer LIMIT 100",
  keywords: CQL_KEYWORDS,
  schemaLabel: "tables",
  supportsCount: false,
  emptySchemaLabel: "no tables in this keyspace",
  sampleStatement: (schema) => {
    const table = schema[0]?.name;
    return table ? `SELECT * FROM ${table}` : "SELECT release_version FROM system.local";
  },
  completionsFor: (schema) => [...CQL_KEYWORDS, ...flattenNames(schema)]
};
