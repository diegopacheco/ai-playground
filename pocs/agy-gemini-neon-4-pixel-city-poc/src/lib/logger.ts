import { config } from "./config";

const correlationId = crypto.randomUUID();
let userId: string | undefined;

export function setUser(id: string | undefined): void {
  userId = id;
}

export function currentCorrelationId(): string {
  return correlationId;
}

type Level = "info" | "warn" | "error";
type Fields = Readonly<Record<string, string | number | boolean | undefined>>;

function write(level: Level, event: string, fields: Fields): void {
  const route = typeof window === "undefined" ? "/" : window.location.pathname;
  const entry = {
    level,
    event,
    correlationId,
    route,
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
