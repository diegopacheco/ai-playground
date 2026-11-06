import { sessionToken, type Identity } from "./identity";
import type { Json, PoppyErrorBody } from "./types";
import { wire } from "./wire";

export const POPPY_VERSION = "0.4";

export class PoppyHttpError extends Error {
  constructor(
    readonly status: number,
    readonly body: PoppyErrorBody,
  ) {
    super(`${body.error}: ${body.error_description ?? ""}`);
  }
}

export type CallOptions = { method?: string; body?: Json; session?: boolean };

export class PoppyClient {
  constructor(
    readonly identity: Identity,
    readonly audience: string,
  ) {}

  async call<T>(url: string, options: CallOptions = {}): Promise<T> {
    const method = options.method ?? "GET";
    const headers: Record<string, string> = {};
    let session: string | undefined;
    if (options.session ?? true) {
      session = await sessionToken(this.identity, { audience: this.audience, method, url });
      headers["Poppy-Session"] = session;
      headers["Poppy-Version"] = POPPY_VERSION;
    }
    const request = options.body === undefined ? undefined : JSON.stringify(options.body);
    if (request) headers["Content-Type"] = "application/json";
    const response = await fetch(url, { method, headers, body: request });
    const text = await response.text();
    wire.add({ method, url, status: response.status, session, request, response: text });
    const parsed = text ? JSON.parse(text) : null;
    if (!response.ok) throw new PoppyHttpError(response.status, parsed as PoppyErrorBody);
    return parsed as T;
  }
}
