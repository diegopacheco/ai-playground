export class FactoryfileError extends Error {
  readonly line: number;
  readonly col: number;

  constructor(line: number, col: number, detail: string) {
    super(`line ${line}, col ${col}: ${detail}`);
    this.name = "FactoryfileError";
    this.line = line;
    this.col = col;
  }
}
