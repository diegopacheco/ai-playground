import { readFileSync } from "node:fs";
import { join } from "node:path";

export const PROMPTS_DIR = join(import.meta.dir, "..", "..", "prompts");

export type PromptParams = Record<string, string>;

export function renderPrompt(template: string, params: PromptParams): string {
  const match = template.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) throw new Error("prompt is missing its --- header");
  const declared = [...match[1].matchAll(/\$([a-z_]+)/g)].map(found => found[1]);
  const missing = declared.filter(name => !(params[name] ?? "").trim());
  if (missing.length > 0) throw new Error(`prompt parameters without value: ${missing.map(name => `$${name}`).join(", ")}`);
  return declared
    .sort((a, b) => b.length - a.length)
    .reduce((body, name) => body.replaceAll(`$${name}`, params[name]), match[2])
    .trim();
}

export function loadPrompt(name: string, params: PromptParams, dir: string = PROMPTS_DIR): string {
  return renderPrompt(readFileSync(join(dir, `${name}.md`), "utf8"), params);
}
