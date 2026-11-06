import { readFileSync } from "node:fs";
import type { Factoryfile } from "./model";
import { parse } from "./parse";

export function load(path: string): Factoryfile {
  return parse(readFileSync(path, "utf8"));
}
