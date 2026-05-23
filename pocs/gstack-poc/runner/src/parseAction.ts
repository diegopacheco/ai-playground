import type { Action, Selector } from "./types.ts";

export type ParseResult =
  | { ok: true; action: Action }
  | { ok: false; error: string };

export function parseAction(raw: string): ParseResult {
  let json: unknown;
  try {
    json = JSON.parse(stripCodeFences(raw));
  } catch (e) {
    return { ok: false, error: `not valid JSON: ${(e as Error).message}` };
  }
  if (!isRecord(json)) {
    return { ok: false, error: "expected a JSON object" };
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
  if (!isRecord(raw)) {
    return { ok: false, error: "selector must be an object" };
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
