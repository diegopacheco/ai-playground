export interface SseMessage {
  event: string;
  data: unknown;
  id?: string;
}

const encoder = new TextEncoder();

export function formatSse(message: SseMessage): Uint8Array {
  const parts: string[] = [];
  if (message.id !== undefined) parts.push(`id: ${message.id}`);
  parts.push(`event: ${message.event}`);
  parts.push(`data: ${JSON.stringify(message.data)}`);
  parts.push("", "");
  return encoder.encode(parts.join("\n"));
}

export function sseHeaders(): HeadersInit {
  return {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  };
}
