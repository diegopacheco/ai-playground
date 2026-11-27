import { config } from "./config";
import { CORRELATION_HEADER, currentCorrelationId, logger } from "./logger";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

function errorMessage(raw: unknown, fallback: string): string {
  if (typeof raw === "object" && raw !== null && "error" in raw && typeof raw.error === "string") {
    return raw.error;
  }
  return fallback;
}

async function request<T>(path: string, parse: (raw: unknown) => T, init?: RequestInit): Promise<T> {
  const res = await fetch(`${config.apiUrl}${path}`, {
    ...init,
    headers: { Accept: "application/json", [CORRELATION_HEADER]: currentCorrelationId(), ...init?.headers },
  });
  const raw: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    const error = new ApiError(res.status, errorMessage(raw, res.statusText));
    logger.warn("api call failed", { path, status: res.status, message: error.message });
    throw error;
  }
  return parse(raw);
}

export const http = {
  get: <T>(path: string, parse: (raw: unknown) => T) => request(path, parse),
  upload: <T>(path: string, body: Blob, headers: Record<string, string>, parse: (raw: unknown) => T) =>
    request(path, parse, { method: "POST", body, headers }),
};
