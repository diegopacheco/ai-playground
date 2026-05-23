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
      const selector = resolveSelector(json);
      if (selector.ok === false) return selector;
      return { ok: true, action: { tool: "click", selector: selector.value, reason } };
    }
    case "type": {
      const selector = resolveSelector(json);
      if (selector.ok === false) return selector;
      const text =
        typeof json.text === "string"
          ? json.text
          : typeof json.value === "string"
          ? json.value
          : typeof json.input === "string"
          ? json.input
          : typeof json.content === "string"
          ? json.content
          : undefined;
      if (text === undefined) {
        return {
          ok: false,
          error: `'type' needs a 'text' string — got: ${truncate(JSON.stringify(json), 200)}`,
        };
      }
      return {
        ok: true,
        action: { tool: "type", selector: selector.value, text, reason },
      };
    }
    case "wait_for": {
      const selector = resolveSelector(json);
      if (selector.ok === false) return selector;
      return {
        ok: true,
        action: { tool: "wait_for", selector: selector.value, reason },
      };
    }
    case "assert_text": {
      let text: string | undefined =
        typeof json.text === "string" ? json.text : undefined;
      if (text === undefined && typeof json.expected === "string") {
        text = json.expected;
      }
      if (text === undefined && typeof json.value === "string") {
        text = json.value;
      }
      if (text === undefined && isRecord(json.selector)) {
        const s = json.selector;
        if (typeof s.text === "string") text = s.text;
        else if (typeof s.name === "string") text = s.name;
      }
      if (text === undefined) {
        return {
          ok: false,
          error: `'assert_text' needs a 'text' string — got: ${truncate(JSON.stringify(json), 200)}`,
        };
      }
      let selector: SelectorParse;
      if (json.selector === undefined || json.selector === null) {
        selector = { ok: true, value: { kind: "text", text } };
      } else {
        selector = parseSelector(json.selector);
      }
      if (selector.ok === false) return selector;
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

function resolveSelector(json: Record<string, unknown>): SelectorParse {
  if (json.selector !== undefined && json.selector !== null) {
    return parseSelector(json.selector);
  }
  const inferred = inferSelectorFromTopLevel(json);
  if (inferred !== null) return { ok: true, value: inferred };
  return {
    ok: false,
    error: `action is missing 'selector' and no fallback fields (text/name/placeholder/label/role) — got: ${truncate(JSON.stringify(json), 200)}`,
  };
}

function inferSelectorFromTopLevel(json: Record<string, unknown>): Selector | null {
  if (typeof json.placeholder === "string") {
    return { kind: "placeholder", text: json.placeholder };
  }
  if (typeof json.label === "string") {
    return { kind: "label", text: json.label };
  }
  if (typeof json.role === "string") {
    const name = typeof json.name === "string" ? json.name : undefined;
    return name !== undefined
      ? { kind: "role", role: json.role, name }
      : { kind: "role", role: json.role };
  }
  if (typeof json.name === "string") {
    return { kind: "role", role: "button", name: json.name };
  }
  if (typeof json.testId === "string") {
    return { kind: "test_id", id: json.testId };
  }
  if (typeof json.text === "string" && json.action !== "type" && json.action !== "assert_text") {
    return { kind: "text", text: json.text };
  }
  if (json.action === "wait_for") {
    return { kind: "role", role: "heading" };
  }
  return null;
}

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
