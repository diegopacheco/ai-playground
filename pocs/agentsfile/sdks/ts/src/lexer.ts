import { AgentsfileError } from "./errors";

export type Punctuation = "=" | "{" | "}" | "[" | "]" | ",";

export type Token =
  | { readonly kind: "identifier"; readonly value: string; readonly line: number; readonly col: number }
  | { readonly kind: "string"; readonly value: string; readonly line: number; readonly col: number }
  | { readonly kind: Punctuation; readonly line: number; readonly col: number }
  | { readonly kind: "eof"; readonly line: number; readonly col: number };

const PUNCTUATION: ReadonlySet<string> = new Set(["=", "{", "}", "[", "]", ","]);
const WHITESPACE: ReadonlySet<string> = new Set([" ", "\t", "\r", "\n"]);
const ESCAPES: Readonly<Record<string, string>> = { '"': '"', "\\": "\\", n: "\n", t: "\t" };

const isIdentifierStart = (c: string): boolean => /^[A-Za-z_]$/.test(c);
const isIdentifierPart = (c: string): boolean => /^[A-Za-z0-9_-]$/.test(c);

export function describe(token: Token): string {
  switch (token.kind) {
    case "identifier":
      return "identifier";
    case "string":
      return "string";
    case "eof":
      return "end of file";
    default:
      return `'${token.kind}'`;
  }
}

export function tokenize(src: string): Token[] {
  const chars = Array.from(src);
  const tokens: Token[] = [];
  let i = 0;
  let line = 1;
  let col = 1;

  const advance = (): string => {
    const c = chars[i] ?? "";
    i += 1;
    if (c === "\n") {
      line += 1;
      col = 1;
    } else {
      col += 1;
    }
    return c;
  };

  const readString = (startLine: number, startCol: number): string => {
    advance();
    let value = "";
    for (;;) {
      const c = chars[i];
      if (c === undefined || c === "\n") {
        throw new AgentsfileError(startLine, startCol, "unterminated string");
      }
      if (c === '"') {
        advance();
        return value;
      }
      if (c === "\\") {
        const escLine = line;
        const escCol = col;
        advance();
        const next = chars[i];
        if (next === undefined || next === "\n") {
          throw new AgentsfileError(startLine, startCol, "unterminated string");
        }
        const decoded = ESCAPES[next];
        if (decoded === undefined) {
          throw new AgentsfileError(escLine, escCol, `invalid escape '\\${next}'`);
        }
        advance();
        value += decoded;
        continue;
      }
      value += advance();
    }
  };

  while (i < chars.length) {
    const c = chars[i] ?? "";
    const startLine = line;
    const startCol = col;
    if (WHITESPACE.has(c)) {
      advance();
    } else if (c === "#") {
      while (i < chars.length && chars[i] !== "\n") {
        advance();
      }
    } else if (PUNCTUATION.has(c)) {
      advance();
      tokens.push({ kind: c as Punctuation, line: startLine, col: startCol });
    } else if (c === '"') {
      tokens.push({ kind: "string", value: readString(startLine, startCol), line: startLine, col: startCol });
    } else if (isIdentifierStart(c)) {
      let value = "";
      while (i < chars.length && isIdentifierPart(chars[i] ?? "")) {
        value += advance();
      }
      tokens.push({ kind: "identifier", value, line: startLine, col: startCol });
    } else {
      throw new AgentsfileError(startLine, startCol, `unexpected character '${c}'`);
    }
  }
  tokens.push({ kind: "eof", line, col });
  return tokens;
}
