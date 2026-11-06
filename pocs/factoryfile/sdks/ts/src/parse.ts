import { tokenize } from "./lexer";
import type { Factoryfile } from "./model";
import { parseStatements } from "./parser";
import { validate } from "./schema";

export function parse(src: string): Factoryfile {
  return validate(parseStatements(tokenize(src)));
}
