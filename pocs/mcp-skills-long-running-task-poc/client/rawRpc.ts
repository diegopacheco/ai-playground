import type { JSONRPCMessage, Transport } from "@modelcontextprotocol/client";

export class RpcError extends Error {
  constructor(readonly code: number, message: string) {
    super(message);
  }
}

type Pending = { readonly resolve: (value: unknown) => void; readonly reject: (error: Error) => void };

const ID_PREFIX = "raw-";

function asRecord(value: unknown): Readonly<Record<string, unknown>> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Readonly<Record<string, unknown>>) : undefined;
}

export class RawDispatcher {
  #next = 0;
  readonly #pending = new Map<string, Pending>();

  constructor(private readonly transport: Transport, private readonly envelope: Readonly<Record<string, unknown>>) {
    const inner = transport.onmessage;
    transport.onmessage = (message, extra) => {
      if (!this.#settle(message)) inner?.(message, extra);
    };
  }

  request(method: string, params: Readonly<Record<string, unknown>>): Promise<unknown> {
    const id = `${ID_PREFIX}${++this.#next}`;
    const message: JSONRPCMessage = { jsonrpc: "2.0", id, method, params: { ...params, _meta: this.envelope } };
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.transport.send(message).catch((err: unknown) => {
        this.#pending.delete(id);
        reject(err instanceof Error ? err : new Error(String(err)));
      });
    });
  }

  #settle(message: JSONRPCMessage): boolean {
    const record = asRecord(message);
    const id = record?.["id"];
    if (typeof id !== "string" || !id.startsWith(ID_PREFIX)) return false;
    const pending = this.#pending.get(id);
    if (!pending) return false;
    this.#pending.delete(id);
    const error = asRecord(record?.["error"]);
    if (error) pending.reject(new RpcError(Number(error["code"]), String(error["message"])));
    else pending.resolve(record?.["result"]);
    return true;
  }
}
