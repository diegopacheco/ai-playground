import type { Action, Selector } from "./types.ts";

export type ParseResult =
  | { ok: true; action: Action }
  | { ok: false; error: string };

export function parseAction(raw: string): ParseResult {
  const candidate = stripCodeFences(raw);
  let json: unknown;
  try {
    json = JSON.parse(candidate);
  } catch {
    const extracted = extractFirstJsonObject(candidate);
    if (extracted === null) {
      return {
        ok: false,
        error: `not valid JSON and no embedded object found in: ${truncate(raw, 200)}`,
      };
    }
    try {
      json = JSON.parse(extracted);
    } catch (e) {
      return {
        ok: false,
        error: `embedded JSON failed to parse: ${(e as Error).message} — raw: ${truncate(raw, 200)}`,
      };
    }
  }
  if (!isRecord(json)) {
    return {
      ok: false,
      error: `expected a JSON object — raw: ${truncate(raw, 200)}`,
    };
  }

  const reason = json.reason;
  if (typeof reason !== "string" || reason.length === 0) {
    return { ok: false, error: "missing or empty 'reason' string" };
  }

  const action = json.action;
  if (typeof action !== "string") {
    return { ok: false, error: "missing 'action' string" };
  }

  switch (action) {
    case "click": {
      const selector = parseSelector(json.selector);
      if (selector.ok === false) return selector;
      return { ok: true, action: { tool: "click", selector: selector.value, reason } };
    }
    case "type": {
      const selector = parseSelector(json.selector);
      if (selector.ok === false) return selector;
      const text = json.text;
      if (typeof text !== "string") return { ok: false, error: "'type' needs a 'text' string" };
      return {
        ok: true,
        action: { tool: "type", selector: selector.value, text, reason },
      };
    }
    case "wait_for": {
      const selector = parseSelector(json.selector);
      if (selector.ok === false) return selector;
      return {
        ok: true,
        action: { tool: "wait_for", selector: selector.value, reason },
      };
    }
    case "assert_text": {
      const selector = parseSelector(json.selector);
      if (selector.ok === false) return selector;
      const text = json.text;
      if (typeof text !== "string")
        return { ok: false, error: "'assert_text' needs a 'text' string" };
      return {
        ok: true,
        action: { tool: "assert_text", selector: selector.value, text, reason },
      };
    }
    case "screenshot":
      return { ok: true, action: { tool: "screenshot", reason } };
    case "done":
      return { ok: true, action: { tool: "done", reason } };
    default:
      return { ok: false, error: `unknown action: ${action}` };
  }
}

type SelectorParse =
  | { ok: true; value: Selector }
  | { ok: false; error: string };

function parseSelector(raw: unknown): SelectorParse {
  if (typeof raw === "string" && raw.length > 0) {
    return { ok: true, value: { kind: "text", text: raw } };
  }
  if (!isRecord(raw)) {
    return {
      ok: false,
      error: `selector must be an object, got ${JSON.stringify(raw) ?? typeof raw}`,
    };
  }
  const kind = raw.kind;
  switch (kind) {
    case "role": {
      if (typeof raw.role !== "string") return { ok: false, error: "role selector needs 'role' string" };
      const name = typeof raw.name === "string" ? raw.name : undefined;
      return {
        ok: true,
        value: name !== undefined
          ? { kind: "role", role: raw.role, name }
          : { kind: "role", role: raw.role },
      };
    }
    case "placeholder":
      if (typeof raw.text !== "string")
        return { ok: false, error: "placeholder selector needs 'text' string" };
      return { ok: true, value: { kind: "placeholder", text: raw.text } };
    case "text":
      if (typeof raw.text !== "string")
        return { ok: false, error: "text selector needs 'text' string" };
      return { ok: true, value: { kind: "text", text: raw.text } };
    case "label":
      if (typeof raw.text !== "string")
        return { ok: false, error: "label selector needs 'text' string" };
      return { ok: true, value: { kind: "label", text: raw.text } };
    case "test_id":
      if (typeof raw.id !== "string")
        return { ok: false, error: "test_id selector needs 'id' string" };
      return { ok: true, value: { kind: "test_id", id: raw.id } };
    case undefined: {
      if (typeof raw.role === "string") {
        const name = typeof raw.name === "string" ? raw.name : undefined;
        return {
          ok: true,
          value: name !== undefined
            ? { kind: "role", role: raw.role, name }
            : { kind: "role", role: raw.role },
        };
      }
      if (typeof raw.placeholder === "string")
        return { ok: true, value: { kind: "placeholder", text: raw.placeholder } };
      if (typeof raw.label === "string")
        return { ok: true, value: { kind: "label", text: raw.label } };
      if (typeof raw.text === "string")
        return { ok: true, value: { kind: "text", text: raw.text } };
      if (typeof raw.testId === "string")
        return { ok: true, value: { kind: "test_id", id: raw.testId } };
      return {
        ok: false,
        error: `selector missing 'kind' and no recognizable field — got ${JSON.stringify(raw)}`,
      };
    }
    default:
      return { ok: false, error: `unknown selector kind: ${String(kind)}` };
  }
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function stripCodeFences(raw: string): string {
  const trimmed = raw.trim();
  const fenceMatch = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
  return fenceMatch && fenceMatch[1] !== undefined ? fenceMatch[1] : trimmed;
}

function extractFirstJsonObject(text: string): string | null {
  const start = text.indexOf("{");
  if (start === -1) return null;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (escape) { escape = false; continue; }
    if (ch === "\\") { escape = true; continue; }
    if (ch === "\"") { inString = !inString; continue; }
    if (inString) continue;
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

function truncate(s: string, max: number): string {
  const collapsed = s.replace(/\s+/g, " ").trim();
  return collapsed.length <= max ? collapsed : collapsed.slice(0, max - 3) + "...";
}
