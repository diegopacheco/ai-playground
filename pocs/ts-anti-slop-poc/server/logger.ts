export const CORRELATION_HEADER = "X-Correlation-Id";

type Level = "info" | "warn" | "error";
type Fields = Readonly<Record<string, string | number | boolean | undefined>>;

export type LogContext = { readonly correlationId: string; readonly path: string; readonly userId: string | undefined };

function write(service: string, level: Level, event: string, context: LogContext, fields: Fields): void {
  const entry = { level, event, service, ...context, ...fields };
  console[level](JSON.stringify(entry));
}

export function createLogger(service: string) {
  return {
    info: (event: string, context: LogContext, fields: Fields = {}) => write(service, "info", event, context, fields),
    error: (event: string, context: LogContext, fields: Fields = {}) => write(service, "error", event, context, fields),
  };
}
