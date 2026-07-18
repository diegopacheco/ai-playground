import type { EngineDescriptor } from "./types";

const COMMANDS = [
  "list topics", "list groups", "describe topic", "describe group", "offsets", "consume",
  "--partition", "--limit", "--from", "earliest", "latest", "offset", "timestamp"
];

export const kafka: EngineDescriptor = {
  kind: "kafka",
  label: "Kafka",
  language: "command",
  placeholder: "consume orders.events --limit 100",
  keywords: COMMANDS,
  schemaLabel: "topics",
  supportsCount: false,
  emptySchemaLabel: "no topics in this cluster",
  sampleStatement: (schema) => {
    const topic = schema[0]?.name;
    return topic ? `consume ${topic} --limit 100` : "list topics";
  },
  completionsFor: (schema) => [...COMMANDS, ...schema.map((node) => node.name)]
};
