export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export type DeliveryMode = "poll" | "push" | "webhook";

export type SchemaProperty = {
  type: string;
  description?: string;
  enum?: string[];
};

export type ObjectSchema = {
  type: "object";
  properties: Record<string, SchemaProperty>;
};

export type EventDescriptor = {
  name: string;
  description: string;
  delivery: DeliveryMode[];
  inputSchema: ObjectSchema;
  payloadSchema: ObjectSchema;
};

export type EventOccurrence = {
  eventId: string;
  name: string;
  timestamp: string;
  data: JsonObject;
  cursor?: string | null;
};

export type PollResult = {
  events: EventOccurrence[];
  cursor: string | null;
  truncated: boolean;
  hasMore: boolean;
  nextPollMs: number;
};

export type SubscriptionMeta = { "io.modelcontextprotocol/subscriptionId": number };

export type StreamNotification =
  | { method: "notifications/events/active"; params: { cursor: string | null; truncated: boolean; _meta: SubscriptionMeta } }
  | { method: "notifications/events/event"; params: EventOccurrence & { _meta: SubscriptionMeta } }
  | { method: "notifications/events/heartbeat"; params: { cursor: string | null; _meta: SubscriptionMeta } }
  | { method: "notifications/events/error"; params: { error: RpcErrorBody; _meta: SubscriptionMeta } }
  | { method: "notifications/events/terminated"; params: { error: RpcErrorBody; _meta: SubscriptionMeta } };

export type RpcErrorBody = { code: number; message: string; data?: JsonValue };

export type SubscriptionRequest = {
  name: string;
  arguments: Record<string, string>;
  cursor: string | null;
  maxAgeMs?: number;
};

export type ToolResult<T> = { structuredContent: T; isError: boolean };

export type DeployResult = {
  deployId: string;
  service: string;
  env: string;
  triggeredBy: string;
  cursor: string;
};

export type ConnectedClient = {
  sessionId: string;
  clientName: string;
  clientVersion: string;
  protocolVersion: string;
  startedAt: string;
  lastSeen: string;
  requests: number;
  closed: boolean;
};
