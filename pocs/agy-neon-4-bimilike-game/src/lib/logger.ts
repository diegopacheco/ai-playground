import { config } from "./config";

export const CORRELATION_HEADER = "X-Correlation-Id";

const correlationId = typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
  ? crypto.randomUUID()
  : `session-${Date.now()}`;

let userId: string | undefined;

export function setUser(id: string | undefined): void {
  userId = id;
}

export function currentCorrelationId(): string {
  return correlationId;
}

type Level = "info" | "warn" | "error";
type Fields = Readonly<Record<string, string | number | boolean | undefined>>;

function currentRoute(): string {
  if (typeof window === "undefined") return "/";
  return window.location.pathname;
}

function write(level: Level, event: string, fields: Fields): void {
  const entry = {
    level,
    event,
    correlationId,
    route: currentRoute(),
    userId,
    app: config.appName,
    ...fields,
  };
  console[level](JSON.stringify(entry));
}

export const logger = {
  info: (event: string, fields: Fields = {}) => write("info", event, fields),
  warn: (event: string, fields: Fields = {}) => write("warn", event, fields),
  error: (event: string, fields: Fields = {}) => write("error", event, fields),
};
