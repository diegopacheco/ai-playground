import { config } from "./config";
import { CORRELATION_HEADER, currentCorrelationId, currentUser, logger } from "./logger";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly body: unknown,
  ) {
    super(message);
  }
}

function errorMessage(body: unknown, status: number): string {
  if (typeof body === "object" && body !== null && "error" in body && typeof body.error === "string") {
    return body.error;
  }
  return `request failed with status ${status}`;
}

async function request<T>(path: string, parse: (raw: unknown) => T, init?: RequestInit): Promise<T> {
  const user = currentUser();
  const res = await fetch(`${config.apiBase}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      [CORRELATION_HEADER]: currentCorrelationId(),
      ...(user ? { "X-User-Id": user } : {}),
      ...init?.headers,
    },
  });
  const text = await res.text();
  const body: unknown = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const message = errorMessage(body, res.status);
    logger.warn("api error", { path, status: res.status, message });
    throw new ApiError(res.status, message, body);
  }
  return parse(body);
}

export const http = {
  get: <T>(path: string, parse: (raw: unknown) => T) => request(path, parse),
  post: <T>(path: string, body: unknown, parse: (raw: unknown) => T) =>
    request(path, parse, { method: "POST", body: JSON.stringify(body) }),
  del: (path: string) => request(path, () => undefined, { method: "DELETE" }),
};
