import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { listen } from "../lib/mcp-server.js";
import { createMathServer } from "../mcp-servers/math-mcp/server.js";
import { createTextServer } from "../mcp-servers/text-mcp/server.js";
import { createGatewayServer } from "../gateway.js";

const servers = [];
let gatewayUrl;
let brokenGatewayUrl;

async function start(server) {
  servers.push(server);
  return `http://127.0.0.1:${await listen(server, 0)}/mcp`;
}

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json, text/event-stream" },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  return { status: response.status, body: text ? JSON.parse(text) : null };
}

const rpc = (url, id, method, params) => post(url, { jsonrpc: "2.0", id, method, params });

before(async () => {
  const mathUrl = await start(createMathServer());
  const textUrl = await start(createTextServer());
  gatewayUrl = await start(createGatewayServer([{ name: "math", url: mathUrl }, { name: "text", url: textUrl }]));
  brokenGatewayUrl = await start(
    createGatewayServer([{ name: "math", url: mathUrl }, { name: "text", url: "http://127.0.0.1:1/mcp" }]),
  );
});

after(() => servers.forEach((server) => server.close()));

test("the gateway answers the MCP handshake as a tools server", async () => {
  const { body } = await rpc(gatewayUrl, 1, "initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "t", version: "1" } });
  assert.equal(body.result.serverInfo.name, "mcp-gateway");
  assert.ok(body.result.capabilities.tools);
  const notification = await post(gatewayUrl, { jsonrpc: "2.0", method: "notifications/initialized" });
  assert.equal(notification.status, 202);
});

test("one tools/list on the gateway exposes the tools of both upstream MCPs", async () => {
  const { body } = await rpc(gatewayUrl, 2, "tools/list", {});
  const names = body.result.tools.map((tool) => tool.name).sort();
  assert.deepEqual(names, ["math__add", "math__multiply", "text__reverse", "text__uppercase", "text__word_count"]);
  assert.ok(body.result.tools.every((tool) => tool.inputSchema?.type === "object"));
});

test("tools/call on the gateway is routed to the right upstream and returns its real result", async () => {
  const sum = await rpc(gatewayUrl, 3, "tools/call", { name: "math__add", arguments: { a: 20, b: 22 } });
  assert.equal(sum.body.result.content[0].text, "42");
  const count = await rpc(gatewayUrl, 4, "tools/call", { name: "text__word_count", arguments: { text: "route me please" } });
  assert.equal(count.body.result.content[0].text, "3");
});

test("tool errors from an upstream reach the client unchanged", async () => {
  const { body } = await rpc(gatewayUrl, 5, "tools/call", { name: "math__multiply", arguments: { a: 1 } });
  assert.equal(body.result.isError, true);
});

test("an upstream that is down drops from tools/list but a call to it fails loud", async () => {
  const list = await rpc(brokenGatewayUrl, 6, "tools/list", {});
  assert.deepEqual(list.body.result.tools.map((tool) => tool.name).sort(), ["math__add", "math__multiply"]);
  const call = await rpc(brokenGatewayUrl, 7, "tools/call", { name: "text__reverse", arguments: { text: "x" } });
  assert.match(call.body.error.message, /upstream unreachable/);
});

test("protocol errors use JSON-RPC codes", async () => {
  const unknownMethod = await rpc(gatewayUrl, 8, "resources/list", {});
  assert.equal(unknownMethod.body.error.code, -32601);
  const unknownTool = await rpc(gatewayUrl, 9, "tools/call", { name: "nope__x", arguments: {} });
  assert.equal(unknownTool.body.error.code, -32602);
  const health = await fetch(gatewayUrl.replace("/mcp", "/health")).then((r) => r.json());
  assert.deepEqual(health, { status: "UP" });
});
