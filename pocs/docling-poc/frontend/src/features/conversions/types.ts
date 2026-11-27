import { ApiError } from "../../http";

export const STATUSES = ["success", "failed"] as const;
export type Status = (typeof STATUSES)[number];

export type Stats = {
  readonly pages: number;
  readonly texts: number;
  readonly tables: number;
  readonly pictures: number;
  readonly headings: number;
  readonly words: number;
};

export type TableView = {
  readonly index: number;
  readonly caption: string;
  readonly headers: readonly string[];
  readonly rows: readonly (readonly string[])[];
};

export type Chunk = {
  readonly index: number;
  readonly headings: readonly string[];
  readonly text: string;
};

export type ConvertedDocument = {
  readonly markdown: string;
  readonly html: string;
  readonly tables: readonly TableView[];
  readonly chunks: readonly Chunk[];
  readonly stats: Stats;
};

export type ConversionSummary = {
  readonly id: string;
  readonly filename: string;
  readonly format: string;
  readonly ocr: boolean;
  readonly status: Status;
  readonly createdAt: string;
  readonly durationMs: number;
  readonly pages: number;
  readonly tables: number;
  readonly error: string | null;
};

export type Conversion = {
  readonly id: string;
  readonly filename: string;
  readonly format: string;
  readonly ocr: boolean;
  readonly status: Status;
  readonly createdAt: string;
  readonly durationMs: number;
  readonly document: ConvertedDocument | null;
  readonly error: string | null;
};

type Obj = Readonly<Record<string, unknown>>;

function invalid(what: string): never {
  throw new ApiError(422, `invalid ${what} payload`);
}

function obj(raw: unknown, what: string): Obj {
  if (typeof raw === "object" && raw !== null && !Array.isArray(raw)) {
    return Object.fromEntries(Object.entries(raw));
  }
  return invalid(what);
}

function str(o: Obj, key: string, what: string): string {
  const v = o[key];
  return typeof v === "string" ? v : invalid(what);
}

function num(o: Obj, key: string, what: string): number {
  const v = o[key];
  return typeof v === "number" ? v : invalid(what);
}

function bool(o: Obj, key: string, what: string): boolean {
  const v = o[key];
  return typeof v === "boolean" ? v : invalid(what);
}

function nullableStr(o: Obj, key: string, what: string): string | null {
  const v = o[key];
  if (v === null) return null;
  return typeof v === "string" ? v : invalid(what);
}

function list<T>(raw: unknown, what: string, item: (v: unknown) => T): readonly T[] {
  return Array.isArray(raw) ? raw.map(item) : invalid(what);
}

function strings(raw: unknown, what: string): readonly string[] {
  return list(raw, what, (v) => (typeof v === "string" ? v : invalid(what)));
}

function status(o: Obj): Status {
  const v = o.status;
  const found = STATUSES.find((s) => s === v);
  return found ?? invalid("status");
}

function parseStats(raw: unknown): Stats {
  const o = obj(raw, "stats");
  return {
    pages: num(o, "pages", "stats"),
    texts: num(o, "texts", "stats"),
    tables: num(o, "tables", "stats"),
    pictures: num(o, "pictures", "stats"),
    headings: num(o, "headings", "stats"),
    words: num(o, "words", "stats"),
  };
}

function parseTable(raw: unknown): TableView {
  const o = obj(raw, "table");
  return {
    index: num(o, "index", "table"),
    caption: str(o, "caption", "table"),
    headers: strings(o.headers, "table"),
    rows: list(o.rows, "table", (r) => strings(r, "table")),
  };
}

function parseChunk(raw: unknown): Chunk {
  const o = obj(raw, "chunk");
  return { index: num(o, "index", "chunk"), headings: strings(o.headings, "chunk"), text: str(o, "text", "chunk") };
}

function parseDocument(raw: unknown): ConvertedDocument | null {
  if (raw === null) return null;
  const o = obj(raw, "document");
  return {
    markdown: str(o, "markdown", "document"),
    html: str(o, "html", "document"),
    tables: list(o.tables, "document", parseTable),
    chunks: list(o.chunks, "document", parseChunk),
    stats: parseStats(o.stats),
  };
}

export function parseConversion(raw: unknown): Conversion {
  const o = obj(raw, "conversion");
  return {
    id: str(o, "id", "conversion"),
    filename: str(o, "filename", "conversion"),
    format: str(o, "format", "conversion"),
    ocr: bool(o, "ocr", "conversion"),
    status: status(o),
    createdAt: str(o, "createdAt", "conversion"),
    durationMs: num(o, "durationMs", "conversion"),
    document: parseDocument(o.document),
    error: nullableStr(o, "error", "conversion"),
  };
}

function parseSummary(raw: unknown): ConversionSummary {
  const o = obj(raw, "summary");
  return {
    id: str(o, "id", "summary"),
    filename: str(o, "filename", "summary"),
    format: str(o, "format", "summary"),
    ocr: bool(o, "ocr", "summary"),
    status: status(o),
    createdAt: str(o, "createdAt", "summary"),
    durationMs: num(o, "durationMs", "summary"),
    pages: num(o, "pages", "summary"),
    tables: num(o, "tables", "summary"),
    error: nullableStr(o, "error", "summary"),
  };
}

export function parseHistory(raw: unknown): readonly ConversionSummary[] {
  return list(raw, "history", parseSummary);
}

export function parseExtensions(raw: unknown): readonly string[] {
  return strings(obj(raw, "formats").extensions, "formats");
}
