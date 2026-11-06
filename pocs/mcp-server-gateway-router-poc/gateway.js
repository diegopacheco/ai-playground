import { createDispatcher, createHttpServer, listen, requirePort } from "./lib/mcp-server.js";
import { createRouter, parseUpstreams } from "./router.js";

export function createGatewayServer(upstreams) {
  return createHttpServer(createDispatcher({ name: "mcp-gateway", version: "1.0.0", ...createRouter(upstreams) }));
}

if (import.meta.main) {
  const upstreams = parseUpstreams(process.env.UPSTREAMS);
  const port = await listen(createGatewayServer(upstreams), requirePort());
  console.log(`mcp-gateway listening on http://localhost:${port}/mcp`);
  upstreams.forEach(({ name, url }) => console.log(`  ${name} -> ${url}`));
}
