import { readFileSync } from "node:fs";
import { tokenize } from "./lexer";
import type { Agentsfile } from "./model";
import { parseStatements } from "./parser";
import { validate } from "./schema";

export { AgentsfileError } from "./errors";
export type { Agent, Agentsfile, Command, Mcp, Profile, Skill, Transport } from "./model";

export function parse(src: string): Agentsfile {
  return validate(parseStatements(tokenize(src)));
}

export function load(path: string): Agentsfile {
  return parse(readFileSync(path, "utf8"));
}
