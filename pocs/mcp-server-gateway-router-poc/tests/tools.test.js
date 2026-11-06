import { test } from "node:test";
import assert from "node:assert/strict";
import { toolRegistry } from "../lib/mcp-server.js";
import { tools as mathTools } from "../mcp-servers/math-mcp/tools.js";
import { tools as textTools } from "../mcp-servers/text-mcp/tools.js";

const math = toolRegistry(mathTools);
const text = toolRegistry(textTools);

test("math tools compute real results, not echoes", async () => {
  assert.equal((await math.callTool("add", { a: 2, b: 3 })).content[0].text, "5");
  assert.equal((await math.callTool("multiply", { a: 4, b: -2 })).content[0].text, "-8");
});

test("math tools report bad input as a tool error so the model can correct itself", async () => {
  const result = await math.callTool("add", { a: "2", b: 3 });
  assert.equal(result.isError, true);
  assert.match(result.content[0].text, /must be numbers/);
});

test("text tools transform the input", async () => {
  assert.equal((await text.callTool("uppercase", { text: "mcp" })).content[0].text, "MCP");
  assert.equal((await text.callTool("reverse", { text: "abc" })).content[0].text, "cba");
  assert.equal((await text.callTool("word_count", { text: "  one two\tthree " })).content[0].text, "3");
  assert.equal((await text.callTool("word_count", { text: "   " })).content[0].text, "0");
});

test("an unknown tool is a protocol error, not a silent empty result", async () => {
  await assert.rejects(math.callTool("divide", { a: 1, b: 1 }), /unknown tool: divide/);
});
