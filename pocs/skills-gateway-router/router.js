import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join, resolve, isAbsolute } from "node:path";
import { parseSkillMd } from "./lib/skill-md.js";

export const SEPARATOR = "__";
export const EXIT = { INVALID: 2, UNREACHABLE: 3 };

const STOP_WORDS = new Set(["the", "and", "for", "with", "this", "that", "use", "when", "from", "into", "please", "can", "you", "what", "how", "are", "its", "out", "all"]);

export class GatewayError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

export function validName(name) {
  return typeof name === "string" && /^[a-z0-9]+(-[a-z0-9]+)*$/.test(name);
}

export function parseSources(spec, root) {
  if (!spec || !spec.trim()) throw new Error("no skill sources, set SKILL_SOURCES or sources.env, i.e. text=upstream-skills/text-skills");
  const sources = spec.split(/[,\n]/).map((entry) => entry.trim()).filter(Boolean).map((entry) => {
    const [name, path] = entry.split("=").map((part) => part?.trim());
    if (!validName(name) || !path) throw new Error(`invalid skill source entry: ${entry}`);
    return { name, dir: isAbsolute(path) ? path : resolve(root, path) };
  });
  const names = sources.map((source) => source.name);
  const duplicate = names.find((name, index) => names.indexOf(name) !== index);
  if (duplicate) throw new Error(`duplicate skill source: ${duplicate}`);
  return sources;
}

export function splitSkillName(qualified) {
  const index = qualified.indexOf(SEPARATOR);
  if (index <= 0) return null;
  return { source: qualified.slice(0, index), skill: qualified.slice(index + SEPARATOR.length) };
}

export function tokens(text) {
  return (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((word) => word.length > 2 && !STOP_WORDS.has(word));
}

function sameWord(a, b) {
  return a === b || (Math.min(a.length, b.length) >= 4 && (a.startsWith(b) || b.startsWith(a)));
}

export function score(request, skill) {
  const words = tokens(`${skill.skill.replaceAll("-", " ")} ${skill.description}`);
  return [...new Set(tokens(request))].filter((word) => words.some((other) => sameWord(word, other))).length;
}

function readSkill(source, folder) {
  const dir = join(source.dir, folder);
  const { meta, body } = parseSkillMd(readFileSync(join(dir, "SKILL.md"), "utf8"));
  const name = meta.name || folder;
  if (!validName(name)) throw new Error(`invalid skill name ${name} in source ${source.name}`);
  if (!meta.description) throw new Error(`skill ${name} in source ${source.name} has no description`);
  return { name: `${source.name}${SEPARATOR}${name}`, source: source.name, skill: name, description: meta.description, dir, body };
}

function scan(source) {
  if (!existsSync(source.dir)) throw new GatewayError(EXIT.UNREACHABLE, `skill source unreachable: ${source.name} at ${source.dir}`);
  const skills = [];
  const errors = [];
  for (const entry of readdirSync(source.dir, { withFileTypes: true }).toSorted((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory() || !existsSync(join(source.dir, entry.name, "SKILL.md"))) continue;
    try {
      skills.push(readSkill(source, entry.name));
    } catch (error) {
      errors.push(error.message);
    }
  }
  return { skills, errors };
}

export function render(skill, args) {
  const body = skill.body.replaceAll("${SKILL_DIR}", skill.dir).replaceAll("$ARGUMENTS", args);
  return `skill: ${skill.name}\nbase directory: ${skill.dir}\n\n${body}\n`;
}

export function createRouter(sources) {
  const byName = new Map(sources.map((source) => [source.name, source]));

  function catalog() {
    const skills = [];
    const errors = [];
    for (const source of byName.values()) {
      try {
        const result = scan(source);
        skills.push(...result.skills);
        errors.push(...result.errors);
      } catch (error) {
        errors.push(error.message);
      }
    }
    return { skills, errors };
  }

  function route(request, limit = 3) {
    if (!request || !tokens(request).length) throw new GatewayError(EXIT.INVALID, "route needs a request with at least one meaningful word");
    const { skills, errors } = catalog();
    const matches = skills
      .map((skill) => ({ ...skill, score: score(request, skill) }))
      .filter((skill) => skill.score > 0)
      .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name))
      .slice(0, limit);
    return { matches, errors };
  }

  function load(qualified, args = "") {
    const parts = splitSkillName(qualified ?? "");
    const source = parts && byName.get(parts.source);
    if (!source) throw new GatewayError(EXIT.INVALID, `no skill source for: ${qualified}`);
    const skill = scan(source).skills.find((candidate) => candidate.skill === parts.skill);
    if (!skill) throw new GatewayError(EXIT.INVALID, `unknown skill: ${qualified}`);
    return render(skill, args);
  }

  function status() {
    return [...byName.values()].map((source) => {
      try {
        return { name: source.name, dir: source.dir, status: "UP", skills: scan(source).skills.length };
      } catch {
        return { name: source.name, dir: source.dir, status: "DOWN", skills: 0 };
      }
    });
  }

  return { catalog, route, load, status };
}
