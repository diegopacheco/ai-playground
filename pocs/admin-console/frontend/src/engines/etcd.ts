import type { EngineDescriptor } from "./types";
import type { SchemaNode } from "@lib/types";

const COMMANDS = ["get", "range", "--prefix", "--limit"];

function paths(nodes: SchemaNode[], prefix = ""): string[] {
  const result: string[] = [];
  for (const node of nodes) {
    const here = `${prefix}/${node.name}`;
    result.push(here);
    if (node.children?.length) {
      result.push(...paths(node.children, here));
    }
  }
  return result;
}

export const etcd: EngineDescriptor = {
  kind: "etcd",
  label: "etcd",
  language: "command",
  placeholder: "get /config --prefix",
  keywords: COMMANDS,
  schemaLabel: "prefixes",
  supportsCount: false,
  emptySchemaLabel: "no keys in this cluster",
  sampleStatement: (schema) => {
    const root = schema[0]?.name;
    return root ? `get /${root} --prefix` : "get / --prefix";
  },
  completionsFor: (schema) => [...COMMANDS, ...paths(schema)]
};
