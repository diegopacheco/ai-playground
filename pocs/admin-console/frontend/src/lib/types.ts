export type ConnectionKind =
  | "mysql"
  | "postgres"
  | "cassandra"
  | "redis"
  | "etcd"
  | "kafka"
  | "elasticsearch";

export interface Connection {
  id: number;
  projectId: number;
  name: string;
  kind: ConnectionKind;
  host: string;
  port: number;
  database?: string | null;
  keyspace?: string | null;
  datacenter?: string | null;
  username?: string | null;
  hasPassword: boolean;
  createdBy: string;
}

export interface Project {
  id: number;
  name: string;
  createdBy: string;
  connections: Connection[];
}

export interface SchemaNode {
  name: string;
  kind: string;
  detail?: string | null;
  children?: SchemaNode[];
}

export interface QueryResult {
  queryId: string;
  columns: string[];
  rows: Record<string, unknown>[];
  elapsedMs: number;
  pageNumber: number;
  nextCursor: string | null;
  hasMore: boolean;
  totalRows: number | null;
}

export interface AiSuggestion {
  statement: string;
  cli: string;
  model: string | null;
  readOnlyOk: boolean;
  denialReason: string | null;
}

export interface AiCli {
  cli: string;
  label: string;
  model: string | null;
  installed: boolean;
  enabled: boolean;
  reason: string | null;
}

export interface Session {
  username: string;
  role: "admin" | "user";
  usingBootstrapPassword?: boolean;
}

export interface AuditEntry {
  id: number;
  queryId: string;
  page: number;
  at: string;
  username: string;
  connectionId: number | null;
  kind: string | null;
  statement: string;
  allowed: boolean;
  denialReason: string | null;
  elapsedMs: number | null;
  rowCount: number | null;
  error: string | null;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly readOnlyViolation = false
  ) {
    super(message);
    this.name = "ApiError";
  }
}
