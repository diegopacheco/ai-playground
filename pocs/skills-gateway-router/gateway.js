#!/usr/bin/env node
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { createRouter, parseSources, GatewayError, EXIT } from "./router.js";

const ROOT = import.meta.dirname;
const USAGE = "usage: gateway.js sources | list | route <request> | load <source__skill> [arguments]";

function readSources() {
  const file = join(ROOT, "sources.env");
  const spec = process.env.SKILL_SOURCES ?? (existsSync(file) ? readFileSync(file, "utf8") : "");
  return parseSources(spec, ROOT);
}

function warn(errors) {
  for (const error of errors) process.stderr.write(`warning: ${error}\n`);
}

function line(skill) {
  return `- ${skill.name}: ${skill.description}`;
}

const commands = {
  sources: (router) => router.status().map((source) => `${source.name} ${source.status} ${source.skills} skills`).join("\n"),
  list: (router) => {
    const { skills, errors } = router.catalog();
    warn(errors);
    return skills.map(line).join("\n") || "no skills found";
  },
  route: (router, args) => {
    const { matches, errors } = router.route(args.join(" "));
    warn(errors);
    return matches.map(line).join("\n") || "no matching skill, run list to see the whole catalog";
  },
  load: (router, [name, ...args]) => router.load(name, args.join(" ")).trimEnd(),
};

export function run(argv) {
  const [command, ...args] = argv;
  const handler = commands[command];
  if (!handler) throw new GatewayError(EXIT.INVALID, USAGE);
  return handler(createRouter(readSources()), args);
}

if (import.meta.main) {
  try {
    process.stdout.write(`${run(process.argv.slice(2))}\n`);
  } catch (error) {
    process.stderr.write(`error: ${error.message}\n`);
    process.exit(error.code ?? 1);
  }
}
