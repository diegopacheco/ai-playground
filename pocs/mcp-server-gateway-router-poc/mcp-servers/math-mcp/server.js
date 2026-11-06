import { createDispatcher, createHttpServer, listen, requirePort, toolRegistry } from "../../lib/mcp-server.js";
import { tools } from "./tools.js";

export function createMathServer() {
  return createHttpServer(createDispatcher({ name: "math-mcp", version: "1.0.0", ...toolRegistry(tools) }));
}

if (import.meta.main) {
  const port = await listen(createMathServer(), requirePort());
  console.log(`math-mcp listening on http://localhost:${port}/mcp`);
}
