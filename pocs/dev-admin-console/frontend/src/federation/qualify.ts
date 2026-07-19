export function aliasFor(statement: string, connectionName: string, source: string): string | null {
  const escaped = `${connectionName}.${source}`.replace(/[.*+?^${}()|[\]\\/:-]/g, "\\$&");
  const match = new RegExp(`${escaped}\\s+(?:AS\\s+)?([a-zA-Z_][\\w]*)`, "i").exec(statement);
  if (!match) {
    return null;
  }
  const candidate = match[1];
  return ["join", "on", "where", "limit", "left", "inner"].includes(candidate.toLowerCase()) ? null : candidate;
}

export function qualify(statement: string, path: string[]): string {
  if (path.length === 1) {
    return path[0];
  }
  if (path.length === 2) {
    return `${path[0]}.${path[1]}`;
  }
  const [connectionName, source, column] = path;
  const alias = aliasFor(statement, connectionName, source);
  return alias ? `${alias}.${column}` : column;
}

export function insertAt(statement: string, text: string): string {
  if (statement.length === 0) {
    return text;
  }
  const needsSpace = !/\s$/.test(statement);
  return `${statement}${needsSpace ? " " : ""}${text}`;
}
