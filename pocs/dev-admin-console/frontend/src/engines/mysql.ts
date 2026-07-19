import type { EngineDescriptor } from "./types";
import { flattenNames } from "./types";

const SQL_KEYWORDS = [
  "SELECT", "FROM", "WHERE", "ORDER BY", "GROUP BY", "HAVING", "LIMIT", "OFFSET", "JOIN",
  "LEFT JOIN", "INNER JOIN", "ON", "AS", "AND", "OR", "NOT", "NULL", "IS", "IN", "LIKE",
  "COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN"
];

export const mysql: EngineDescriptor = {
  kind: "mysql",
  label: "MySQL",
  language: "sql",
  placeholder: "SELECT * FROM invoices ORDER BY id",
  keywords: SQL_KEYWORDS,
  schemaLabel: "tables",
  supportsCount: true,
  emptySchemaLabel: "no tables in this database",
  sampleStatement: (schema) => {
    const table = schema[0]?.name;
    return table ? `SELECT * FROM ${table}` : "SELECT 1";
  },
  completionsFor: (schema) => [...SQL_KEYWORDS, ...flattenNames(schema)]
};
