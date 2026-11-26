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

function problemDetail(raw: string): string {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed === "object" && parsed !== null && "detail" in parsed && typeof parsed.detail === "string") {
      return parsed.detail;
    }
  } catch {
    return raw;
  }
  return raw;
}

async function request<T>(path: string, parse: (raw: unknown) => T, init: RequestInit): Promise<T> {
  const res = await fetch(`${config.apiUrl}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", [CORRELATION_HEADER]: currentCorrelationId(), ...init.headers },
  });
  if (!res.ok) {
    const error = new ApiError(res.status, problemDetail(await res.text()));
    logger.error("api call failed", { path, status: res.status });
    throw error;
  }
  return parse(await res.json());
}

export const http = {
  post: <T>(path: string, body: unknown, parse: (raw: unknown) => T) =>
    request(path, parse, { method: "POST", body: JSON.stringify(body) }),
};
