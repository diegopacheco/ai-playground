import { preview } from "./dbos.ts";
import type { Stored } from "./types.ts";

export function StoredView({ stored, empty = "none" }: { stored: Stored; empty?: string }) {
  const text = preview(stored);
  if (!stored.readable) return <p className="muted">{text}</p>;
  return text ? <pre className="value">{text}</pre> : <p className="muted">{empty}</p>;
}
