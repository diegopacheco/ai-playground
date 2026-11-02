import { parseSse } from "./sse";
import type { JsonObject, JsonValue, RpcErrorBody, StreamNotification, SubscriptionRequest } from "./types";

const PROTOCOL_VERSION = "2025-11-25";
const SESSION_HEADER = "mcp-session-id";

export class McpError extends Error {
  constructor(readonly body: RpcErrorBody) {
    super(`${body.code} ${body.message}`);
  }
}

type RpcResponse = { id: number; result?: JsonValue; error?: RpcErrorBody };

export class McpClient {
  private sessionId: string | null = null;
  private nextId = 1;
  private ready: Promise<void> | null = null;

  constructor(
    private readonly endpoint: string,
    private readonly clientName: string,
  ) {}

  connect(): Promise<void> {
    this.ready ??= this.initialize();
    return this.ready;
  }

  async request<T>(method: string, params: JsonObject = {}): Promise<T> {
    await this.connect();
    return this.send<T>(method, params);
  }

  async stream(request: SubscriptionRequest, onMessage: (message: StreamNotification) => void, signal: AbortSignal): Promise<number> {
    await this.connect();
    const id = this.nextId++;
    const response = await fetch(this.endpoint, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ jsonrpc: "2.0", id, method: "events/stream", params: request }),
      signal,
    });
    if (response.headers.get("content-type")?.includes("application/json")) {
      const body = (await response.json()) as RpcResponse;
      throw new McpError(body.error ?? { code: -32603, message: "stream did not open" });
    }
    if (!response.body) throw new McpError({ code: -32603, message: "stream has no body" });
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) return id;
      const chunk = parseSse(buffer + value);
      buffer = chunk.rest;
      chunk.messages.forEach((data) => onMessage(JSON.parse(data) as StreamNotification));
    }
  }

  private async initialize(): Promise<void> {
    const response = await fetch(this.endpoint, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: this.nextId++,
        method: "initialize",
        params: { protocolVersion: PROTOCOL_VERSION, capabilities: {}, clientInfo: { name: this.clientName, version: "0.1.0" } },
      }),
    });
    this.sessionId = response.headers.get(SESSION_HEADER);
    await fetch(this.endpoint, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
    });
  }

  private async send<T>(method: string, params: JsonObject): Promise<T> {
    const response = await fetch(this.endpoint, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ jsonrpc: "2.0", id: this.nextId++, method, params }),
    });
    const body = (await response.json()) as RpcResponse;
    if (body.error) throw new McpError(body.error);
    return body.result as T;
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = { "content-type": "application/json", accept: "application/json, text/event-stream" };
    if (this.sessionId) headers[SESSION_HEADER] = this.sessionId;
    return headers;
  }
}

export const mcp = new McpClient("/mcp", "mcp-events-ui");
