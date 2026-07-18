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

export interface TraceHit {
  connectionName: string;
  kind: string;
  source: string;
  label: string;
  at: string | null;
  columns: string[];
  row: Record<string, unknown>;
}

export interface TraceResult {
  term: string;
  elapsedMs: number;
  truncated: boolean;
  hits: TraceHit[];
  failures: { connectionName: string; kind: string; reason: string }[];
}

export interface FederatedSide {
  alias: string;
  connectionName: string;
  kind: string;
  source: string;
  rows: number;
  truncated: boolean;
}

export interface FederatedResult {
  columns: string[];
  rows: Record<string, unknown>[];
  elapsedMs: number;
  sides: FederatedSide[];
  diagnostic: string | null;
}

export interface DiscoveredContainer {
  id: string;
  name: string;
  image: string;
  kind: string;
  hostPort: number;
  containerPort: number;
  database: string | null;
  keyspace: string | null;
  username: string | null;
  hasPassword: boolean;
  importable: boolean;
  reason: string | null;
}

export interface SavedQuery {
  id: number;
  projectId: number;
  connectionId: number | null;
  name: string;
  statement: string;
  kind: string;
  description: string | null;
  createdBy: string;
  updatedAt: string;
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
