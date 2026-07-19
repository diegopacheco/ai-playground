import type { ConnectionKind, SchemaNode } from "@lib/types";

export interface EngineDescriptor {
  kind: ConnectionKind;
  label: string;
  language: "sql" | "json" | "command";
  placeholder: string;
  keywords: string[];
  emptySchemaLabel: string;
  schemaLabel: string;
  supportsCount: boolean;
  sampleStatement: (schema: SchemaNode[]) => string;
  completionsFor: (schema: SchemaNode[]) => string[];
}

export function flattenNames(nodes: SchemaNode[], depth = 2): string[] {
  const names: string[] = [];
  const walk = (list: SchemaNode[], level: number) => {
    for (const node of list) {
      names.push(node.name);
      if (node.children && level < depth) {
        walk(node.children, level + 1);
      }
    }
  };
  walk(nodes, 0);
  return [...new Set(names)];
}
