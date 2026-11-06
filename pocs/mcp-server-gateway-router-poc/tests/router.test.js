import { test } from "node:test";
import assert from "node:assert/strict";
import { createRouter, parseUpstreams, splitToolName } from "../router.js";

const upstreams = [
  { name: "math", url: "http://math/mcp" },
  { name: "text", url: "http://text/mcp" },
];

function fakeCall(calls, catalog) {
  return async (url, method, params) => {
    calls.push({ url, method, params });
    if (!(url in catalog)) throw new Error(`down: ${url}`);
    if (method === "tools/list") return { tools: catalog[url] };
    return { content: [{ type: "text", text: `${url}:${params.name}` }] };
  };
}

test("parseUpstreams reads name=url pairs and rejects names that would break routing", () => {
  assert.deepEqual(parseUpstreams("math=http://a/mcp, text=http://b/mcp"), [
    { name: "math", url: "http://a/mcp" },
    { name: "text", url: "http://b/mcp" },
  ]);
  assert.throws(() => parseUpstreams(""), /UPSTREAMS env var is required/);
  assert.throws(() => parseUpstreams("bad__name=http://a/mcp"), /invalid upstream entry/);
  assert.throws(() => parseUpstreams("math"), /invalid upstream entry/);
});

test("splitToolName splits on the first separator so tool names with underscores survive", () => {
  assert.deepEqual(splitToolName("text__word_count"), { upstream: "text", tool: "word_count" });
  assert.equal(splitToolName("word_count"), null);
  assert.equal(splitToolName("__add"), null);
});

test("listTools prefixes every tool with its upstream so equal names never collide", async () => {
  const catalog = { "http://math/mcp": [{ name: "echo" }], "http://text/mcp": [{ name: "echo" }] };
  const router = createRouter(upstreams, fakeCall([], catalog));
  const names = (await router.listTools()).map((tool) => tool.name);
  assert.deepEqual(names, ["math__echo", "text__echo"]);
});

test("listTools keeps the healthy upstreams when one is down", async () => {
  const catalog = { "http://text/mcp": [{ name: "reverse", description: "Reverses" }] };
  const router = createRouter(upstreams, fakeCall([], catalog));
  assert.deepEqual(await router.listTools(), [{ name: "text__reverse", description: "[text] Reverses" }]);
});

test("callTool sends the original tool name to the upstream that owns the prefix", async () => {
  const calls = [];
  const catalog = { "http://math/mcp": [], "http://text/mcp": [] };
  const router = createRouter(upstreams, fakeCall(calls, catalog));
  const result = await router.callTool("text__reverse", { text: "ab" });
  assert.equal(result.content[0].text, "http://text/mcp:reverse");
  assert.deepEqual(calls, [{ url: "http://text/mcp", method: "tools/call", params: { name: "reverse", arguments: { text: "ab" } } }]);
});

test("callTool fails loud for an unknown upstream or a name without prefix", async () => {
  const calls = [];
  const router = createRouter(upstreams, fakeCall(calls, {}));
  await assert.rejects(router.callTool("weather__today", {}), /no upstream for tool: weather__today/);
  await assert.rejects(router.callTool("add", {}), /no upstream for tool: add/);
  assert.equal(calls.length, 0);
});
