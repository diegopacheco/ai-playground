import { createDispatcher, createHttpServer, listen, readBody, requirePort, sendJson } from "./lib/mcp-server.js";
import { codeUpstream, markdownUpstream, syncDir, watchDir } from "./lib/loader.js";
import { createRouter, parseUpstreams } from "./router.js";

async function upstreamFrom(body) {
  const { name, url, markdown, module } = body;
  if (typeof url === "string") return { name, url, kind: "url", source: "api" };
  if (typeof markdown === "string") return markdownUpstream(name, markdown, "api");
  if (typeof module === "string") return codeUpstream(name, module, "api");
  throw new Error("body needs name and one of url, markdown or module");
}

function serverRoutes(router) {
  return async (req, res) => {
    const match = req.url.match(/^\/servers(?:\/([^/?]+))?$/);
    if (!match) return false;
    const name = match[1] && decodeURIComponent(match[1]);
    if (req.method === "GET" && !name) return sendJson(res, 200, { servers: router.servers() }), true;
    if (req.method === "DELETE" && name) {
      return router.remove(name) ? (res.writeHead(204), res.end()) : sendJson(res, 404, { error: `no server: ${name}` }), true;
    }
    if (req.method === "POST" && !name) {
      try {
        const upstream = await upstreamFrom(JSON.parse(await readBody(req)));
        router.add(upstream);
        const { local, ...info } = upstream;
        return sendJson(res, 201, { ...info, tools: local ? local.listTools().map((tool) => tool.name) : undefined }), true;
      } catch (error) {
        return sendJson(res, 400, { error: error.message }), true;
      }
    }
    return sendJson(res, 405, { error: "method not allowed" }), true;
  };
}

export function createGatewayServer(upstreams, { dir } = {}) {
  const router = createRouter(upstreams.map((upstream) => ({ ...upstream, kind: "url", source: "env" })));
  const server = createHttpServer(
    createDispatcher({ name: "mcp-gateway", version: "1.1.0", listTools: router.listTools, callTool: router.callTool }),
    serverRoutes(router),
  );
  server.router = router;
  server.ready = Promise.resolve();
  if (dir) {
    server.ready = syncDir(dir, router);
    const watcher = watchDir(dir, router);
    server.watcher = watcher;
    server.on("close", watcher.close);
  }
  return server;
}

if (import.meta.main) {
  const upstreams = process.env.UPSTREAMS ? parseUpstreams(process.env.UPSTREAMS) : [];
  const server = createGatewayServer(upstreams, { dir: process.env.MCP_DIR });
  await server.ready;
  const port = await listen(server, requirePort());
  console.log(`mcp-gateway listening on http://localhost:${port}/mcp`);
  server.router.servers().forEach(({ name, kind, url, source }) => console.log(`  ${name} [${kind}, ${source}] ${url ?? ""}`));
}
