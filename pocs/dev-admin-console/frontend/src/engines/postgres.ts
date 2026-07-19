import type { EngineDescriptor } from "./types";
import { flattenNames } from "./types";

const SQL_KEYWORDS = [
  "SELECT", "FROM", "WHERE", "ORDER BY", "GROUP BY", "HAVING", "LIMIT", "OFFSET", "JOIN",
  "LEFT JOIN", "INNER JOIN", "ON", "AS", "AND", "OR", "NOT", "NULL", "IS", "IN", "LIKE",
  "COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT", "WITH", "EXPLAIN"
];

export const postgres: EngineDescriptor = {
  kind: "postgres",
  label: "Postgres",
  language: "sql",
  placeholder: "SELECT * FROM customers ORDER BY id",
  keywords: SQL_KEYWORDS,
  schemaLabel: "tables",
  supportsCount: true,
  emptySchemaLabel: "no tables in this schema",
  sampleStatement: (schema) => {
    const table = schema[0]?.name;
    return table ? `SELECT * FROM ${table}` : "SELECT 1";
  },
  completionsFor: (schema) => [...SQL_KEYWORDS, ...flattenNames(schema)]
};
