import { config } from "./config.ts";
import { CORRELATION_HEADER, currentCorrelationId, logger } from "./logger.ts";

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, parse: (raw: unknown) => T): Promise<T> {
  const res = await fetch(`${config.apiUrl}${path}`, { headers: { [CORRELATION_HEADER]: currentCorrelationId() } });
  if (!res.ok) {
    logger.error("api call failed", { path, status: res.status });
    throw new ApiError(res.status, await res.text());
  }

  return parse(await res.json());
}

export const http = {
  get: <T,>(path: string, parse: (raw: unknown) => T) => request(path, parse),
};
