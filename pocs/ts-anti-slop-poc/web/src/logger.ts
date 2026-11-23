import { config } from "./config.ts";

export const CORRELATION_HEADER = "X-Correlation-Id";

const correlationId = crypto.randomUUID();

export function currentCorrelationId(): string {
  return correlationId;
}

type Level = "info" | "warn" | "error";
type Fields = Readonly<Record<string, string | number | boolean | undefined>>;

function write(level: Level, event: string, fields: Fields): void {
  const entry = { level, event, correlationId, route: window.location.pathname, userId: undefined, app: config.appName, ...fields };
  console[level](JSON.stringify(entry));
}

export const logger = {
  info: (event: string, fields: Fields = {}) => write("info", event, fields),
  error: (event: string, fields: Fields = {}) => write("error", event, fields),
};
