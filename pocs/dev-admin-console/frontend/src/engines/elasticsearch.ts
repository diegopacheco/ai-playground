import type { EngineDescriptor } from "./types";
import { flattenNames } from "./types";

const COMMANDS = [
  "GET", "HEAD", "_search", "_count", "_mapping", "_cat/indices", "_msearch",
  "query", "match", "match_all", "term", "terms", "range", "bool", "must", "should", "filter", "aggs"
];

export const elasticsearch: EngineDescriptor = {
  kind: "elasticsearch",
  label: "Elasticsearch",
  language: "json",
  placeholder: 'GET /products/_search {"query":{"match_all":{}}}',
  keywords: COMMANDS,
  schemaLabel: "indices",
  supportsCount: false,
  emptySchemaLabel: "no indices in this cluster",
  sampleStatement: (schema) => {
    const index = schema[0]?.name;
    return index ? `GET /${index}/_search` : "GET /_cat/indices?format=json";
  },
  completionsFor: (schema) => [...COMMANDS, ...flattenNames(schema)]
};
