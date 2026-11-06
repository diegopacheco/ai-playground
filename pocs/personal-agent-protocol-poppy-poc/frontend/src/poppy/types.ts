export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type JsonObject = { [key: string]: Json };

export type EcJwk = { kty: "EC"; crv: "P-256"; x: string; y: string; kid?: string };

export type PoppyDocument = {
  protocol_version: string;
  organization: { name: string; domain: string };
  agent?: { protocols: { type: string; endpoint: string }[] };
  apis?: { type: string; url: string; description?: string }[];
  feedback?: { endpoint: string };
  excluded_tasks?: { url: string };
};

export type ExcludedTasks = { tasks: { task: string; alternative?: { instructions: string } }[] };

export type Room = { id: string; name: string; beds: number; price_per_night: number; pets_allowed: boolean };
export type Hold = { id: string; room_id: string; room_name: string; nights: number; total: number };

export type ConversationSummary = { conversation_id: string; status: string; responder: string };
export type ChatMessage = { id: string; role: "user" | "company"; author: { type: string }; text: string };
export type ConversationEvent = { id: string; type: string; message?: ChatMessage; status?: string };
export type EventsPage = ConversationSummary & { events: ConversationEvent[]; cursor: string };

export type Check = { name: string; level: "pass" | "warn" | "fail"; detail: string };
export type InspectorReport = { host: string; valid: boolean; checks: Check[] };

export type PoppyErrorBody = { error: string; error_description?: string };
