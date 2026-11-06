import { AgentsfileError } from "./errors";
import { describe, type Token } from "./lexer";

export interface Position {
  readonly line: number;
  readonly col: number;
}

export interface StringValue extends Position {
  readonly kind: "string";
  readonly value: string;
}

export interface ListValue extends Position {
  readonly kind: "list";
  readonly items: readonly StringValue[];
}

export interface MapEntry extends Position {
  readonly key: string;
  readonly value: StringValue;
}

export interface MapValue extends Position {
  readonly kind: "map";
  readonly entries: readonly MapEntry[];
}

export type Value = StringValue | ListValue | MapValue;

export interface Attribute extends Position {
  readonly type: "attribute";
  readonly name: string;
  readonly value: Value;
}

export interface Block extends Position {
  readonly type: "block";
  readonly keyword: string;
  readonly label: string;
  readonly body: readonly Statement[];
}

export type Statement = Attribute | Block;

export function parseStatements(tokens: readonly Token[]): Statement[] {
  let i = 0;

  const peek = (): Token => tokens[i] ?? tokens[tokens.length - 1]!;
  const next = (): Token => {
    const token = peek();
    if (token.kind !== "eof") {
      i += 1;
    }
    return token;
  };
  const fail = (token: Token, expected: string): never => {
    throw new AgentsfileError(token.line, token.col, `expected ${expected}, found ${describe(token)}`);
  };
  const expectString = (): StringValue => {
    const token = next();
    if (token.kind !== "string") {
      return fail(token, "string");
    }
    return { kind: "string", value: token.value, line: token.line, col: token.col };
  };

  const parseList = (open: Token): ListValue => {
    const items: StringValue[] = [];
    while (peek().kind !== "]") {
      items.push(expectString());
      const separator = next();
      if (separator.kind === "]") {
        return { kind: "list", items, line: open.line, col: open.col };
      }
      if (separator.kind !== ",") {
        fail(separator, "',' or ']'");
      }
    }
    next();
    return { kind: "list", items, line: open.line, col: open.col };
  };

  const parseMap = (open: Token): MapValue => {
    const entries: MapEntry[] = [];
    const keys = new Set<string>();
    for (;;) {
      const key = next();
      if (key.kind === "}") {
        return { kind: "map", entries, line: open.line, col: open.col };
      }
      if (key.kind !== "identifier") {
        return fail(key, "identifier or '}'");
      }
      if (keys.has(key.value)) {
        throw new AgentsfileError(key.line, key.col, `duplicate key "${key.value}"`);
      }
      keys.add(key.value);
      const equals = next();
      if (equals.kind !== "=") {
        fail(equals, "'='");
      }
      entries.push({ key: key.value, value: expectString(), line: key.line, col: key.col });
      if (peek().kind === ",") {
        next();
      }
    }
  };

  const parseValue = (): Value => {
    const token = next();
    switch (token.kind) {
      case "string":
        return { kind: "string", value: token.value, line: token.line, col: token.col };
      case "[":
        return parseList(token);
      case "{":
        return parseMap(token);
      default:
        return fail(token, "value");
    }
  };

  const parseStatement = (name: Token & { kind: "identifier" }): Statement => {
    const token = next();
    if (token.kind === "=") {
      return { type: "attribute", name: name.value, value: parseValue(), line: name.line, col: name.col };
    }
    if (token.kind !== "string") {
      return fail(token, "'=' or block label");
    }
    const open = next();
    if (open.kind !== "{") {
      fail(open, "'{'");
    }
    const body: Statement[] = [];
    for (;;) {
      const start = next();
      if (start.kind === "}") {
        return { type: "block", keyword: name.value, label: token.value, body, line: name.line, col: name.col };
      }
      if (start.kind !== "identifier") {
        return fail(start, "identifier or '}'");
      }
      body.push(parseStatement(start));
    }
  };

  const statements: Statement[] = [];
  while (peek().kind !== "eof") {
    const start = next();
    if (start.kind !== "identifier") {
      return fail(start, "identifier");
    }
    statements.push(parseStatement(start));
  }
  return statements;
}
