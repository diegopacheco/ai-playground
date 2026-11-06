import http from "node:http";

export const PROTOCOL_VERSION = "2025-06-18";

export const ERRORS = {
  PARSE: -32700,
  INVALID_REQUEST: -32600,
  METHOD_NOT_FOUND: -32601,
  INVALID_PARAMS: -32602,
  INTERNAL: -32603,
};

export class McpError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

export function textResult(text) {
  return { content: [{ type: "text", text: String(text) }] };
}

function success(id, result) {
  return { jsonrpc: "2.0", id, result };
}

function failure(id, code, message) {
  return { jsonrpc: "2.0", id: id ?? null, error: { code, message } };
}

export function createDispatcher({ name, version, listTools, callTool }) {
  const methods = {
    initialize: (params) => ({
      protocolVersion: params?.protocolVersion ?? PROTOCOL_VERSION,
      capabilities: { tools: { listChanged: false } },
      serverInfo: { name, version },
    }),
    ping: () => ({}),
    "tools/list": async () => ({ tools: await listTools() }),
    "tools/call": async (params) => {
      if (!params || typeof params.name !== "string") {
        throw new McpError(ERRORS.INVALID_PARAMS, "tools/call requires a string 'name'");
      }
      return callTool(params.name, params.arguments ?? {});
    },
  };

  return async function dispatch(message) {
    if (!message || message.jsonrpc !== "2.0" || typeof message.method !== "string") {
      return failure(message?.id, ERRORS.INVALID_REQUEST, "invalid JSON-RPC 2.0 request");
    }
    const isNotification = message.id === undefined;
    if (isNotification) return null;
    const method = methods[message.method];
    if (!method) {
      return failure(message.id, ERRORS.METHOD_NOT_FOUND, `method not found: ${message.method}`);
    }
    try {
      return success(message.id, await method(message.params));
    } catch (error) {
      const code = error instanceof McpError ? error.code : ERRORS.INTERNAL;
      return failure(message.id, code, error.message);
    }
  };
}

export function toolRegistry(tools) {
  const byName = new Map(tools.map((tool) => [tool.name, tool]));
  return {
    listTools: () => tools.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })),
    callTool: async (name, args) => {
      const tool = byName.get(name);
      if (!tool) throw new McpError(ERRORS.INVALID_PARAMS, `unknown tool: ${name}`);
      try {
        return textResult(await tool.handler(args));
      } catch (error) {
        return { ...textResult(error.message), isError: true };
      }
    },
  };
}

function sendJson(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => (data += chunk));
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });
}

export function createHttpServer(dispatch) {
  return http.createServer(async (req, res) => {
    if (req.method === "GET" && req.url === "/health") return sendJson(res, 200, { status: "UP" });
    if (req.url !== "/mcp") return sendJson(res, 404, { error: "not found" });
    if (req.method !== "POST") return sendJson(res, 405, { error: "method not allowed" });

    let message;
    try {
      message = JSON.parse(await readBody(req));
    } catch {
      return sendJson(res, 400, failure(null, ERRORS.PARSE, "parse error"));
    }
    const response = await dispatch(message);
    if (response === null) {
      res.writeHead(202);
      return res.end();
    }
    sendJson(res, 200, response);
  });
}

export function listen(server, port) {
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve(server.address().port)));
}

export function requirePort() {
  const port = Number(process.env.PORT);
  if (!Number.isInteger(port) || port <= 0) {
    console.error("PORT env var is required");
    process.exit(1);
  }
  return port;
}
