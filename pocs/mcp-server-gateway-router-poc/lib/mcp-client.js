import { McpError, ERRORS, PROTOCOL_VERSION } from "./mcp-server.js";

let nextId = 1;

export async function rpc(url, method, params) {
  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL_VERSION,
      },
      body: JSON.stringify({ jsonrpc: "2.0", id: nextId++, method, params }),
      signal: AbortSignal.timeout(5000),
    });
  } catch (error) {
    throw new McpError(ERRORS.INTERNAL, `upstream unreachable at ${url}: ${error.message}`);
  }
  const body = await response.json();
  if (body.error) throw new McpError(body.error.code, body.error.message);
  return body.result;
}
