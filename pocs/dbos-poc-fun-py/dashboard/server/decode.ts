import type { Json, Stored } from "../src/types.ts";

const JSON_FORMATS = new Set(["agent_json", "portable_json"]);

export function decodeStored(text: string | null, format: string | null): Stored {
  const name = format ?? "none";
  if (text === null) return { format: name, readable: true, value: null };
  if (!JSON_FORMATS.has(name)) return { format: name, readable: false, value: null };
  try {
    return { format: name, readable: true, value: JSON.parse(text) as Json };
  } catch {
    return { format: name, readable: false, value: null };
  }
}
