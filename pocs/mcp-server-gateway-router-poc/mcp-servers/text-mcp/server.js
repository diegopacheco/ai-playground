import { createDispatcher, createHttpServer, listen, requirePort, toolRegistry } from "../../lib/mcp-server.js";
import { tools } from "./tools.js";

export function createTextServer() {
  return createHttpServer(createDispatcher({ name: "text-mcp", version: "1.0.0", ...toolRegistry(tools) }));
}

if (import.meta.main) {
  const port = await listen(createTextServer(), requirePort());
  console.log(`text-mcp listening on http://localhost:${port}/mcp`);
}
