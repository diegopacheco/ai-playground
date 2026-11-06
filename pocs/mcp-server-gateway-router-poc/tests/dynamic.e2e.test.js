import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { listen } from "../lib/mcp-server.js";
import { createTextServer } from "../mcp-servers/text-mcp/server.js";
import { createGatewayServer } from "../gateway.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, "..");
const servers = [];
let dir;
let base;
let textUrl;

async function start(server) {
  servers.push(server);
  return `http://127.0.0.1:${await listen(server, 0)}`;
}

async function rpc(method, params) {
  const response = await fetch(`${base}/mcp`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
  });
  return (await response.json()).result;
}

const toolNames = async () => (await rpc("tools/list", {})).tools.map((tool) => tool.name).sort();

async function api(method, pathname, body) {
  const response = await fetch(`${base}${pathname}`, { method, body: body && JSON.stringify(body) });
  const text = await response.text();
  return { status: response.status, body: text ? JSON.parse(text) : null };
}

async function eventually(check) {
  for (let i = 0; i < 50; i++) {
    if (await check()) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  assert.fail("condition never became true");
}

before(async () => {
  process.env.CLAUDE_BIN = path.join(here, "fixtures", "fake-claude.sh");
  dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "mcp-dir-")));
  fs.cpSync(path.join(root, "dynamic-mcps"), dir, { recursive: true });
  textUrl = `${await start(createTextServer())}/mcp`;
  const gateway = createGatewayServer([{ name: "text", url: textUrl }], { dir });
  await gateway.ready;
  base = await start(gateway);
});

after(() => {
  servers.forEach((server) => server.close());
  fs.rmSync(dir, { recursive: true, force: true });
});

test("servers already in the folder are listed next to the env upstreams at startup", async () => {
  const names = await toolNames();
  for (const name of ["hash__sha256", "writer__summarize", "writer__translate", "text__reverse"]) assert.ok(names.includes(name), name);
});

test("a code server dropped into the folder while the gateway runs is listed and callable with no restart", async () => {
  fs.mkdirSync(path.join(dir, "greet-mcp"));
  fs.writeFileSync(
    path.join(dir, "greet-mcp", "tools.js"),
    `export const tools = [{ name: "hello", description: "Greets", inputSchema: { type: "object", properties: {} }, handler: () => "hi there" }];`,
  );
  await eventually(async () => (await toolNames()).includes("greet__hello"));
  const result = await rpc("tools/call", { name: "greet__hello", arguments: {} });
  assert.equal(result.content[0].text, "hi there");
});

test("an English .md server dropped into the folder is listed, and its calls run through claude -p", async () => {
  fs.writeFileSync(path.join(dir, "poet.md"), "## haiku\nWrites a haiku.\n- topic (string, required): what it is about\n\nWrite a haiku about the topic.\n");
  await eventually(async () => (await toolNames()).includes("poet__haiku"));
  const result = await rpc("tools/call", { name: "poet__haiku", arguments: { topic: "routers" } });
  assert.notEqual(result.isError, true);
  const text = result.content[0].text;
  assert.match(text, /argv=-p --strict-mcp-config/);
  assert.match(text, /Write a haiku about the topic\./);
  assert.match(text, /"topic":"routers"/);
});

test("removing a server file from the folder removes its tools", async () => {
  fs.rmSync(path.join(dir, "poet.md"));
  await eventually(async () => !(await toolNames()).includes("poet__haiku"));
  assert.ok((await toolNames()).includes("hash__sha256"));
});

test("POST /servers registers a markdown, a code module and a remote url server, DELETE drops them", async () => {
  const md = await api("POST", "/servers", { name: "notes", markdown: "## tidy\nTidies a note.\n- note (string, required): the note\n" });
  assert.equal(md.status, 201);
  assert.deepEqual(md.body.tools, ["tidy"]);
  const code = await api("POST", "/servers", { name: "math", module: path.join(root, "mcp-servers", "math-mcp", "tools.js") });
  assert.equal(code.status, 201);
  const remote = await api("POST", "/servers", { name: "text2", url: textUrl });
  assert.equal(remote.status, 201);

  const sum = await rpc("tools/call", { name: "math__add", arguments: { a: 40, b: 2 } });
  assert.equal(sum.content[0].text, "42");
  const reversed = await rpc("tools/call", { name: "text2__reverse", arguments: { text: "abc" } });
  assert.equal(reversed.content[0].text, "cba");
  const listed = (await api("GET", "/servers")).body.servers.map((s) => `${s.name}:${s.kind}:${s.source}`);
  for (const entry of ["notes:md:api", "math:code:api", "text2:url:api", "text:url:env", "hash:code:dir"]) assert.ok(listed.includes(entry), entry);

  assert.equal((await api("DELETE", "/servers/notes")).status, 204);
  assert.equal((await api("DELETE", "/servers/notes")).status, 404);
  assert.ok(!(await toolNames()).includes("notes__tidy"));
});

test("registration rejects names that would break routing and bodies with no server", async () => {
  const badName = await api("POST", "/servers", { name: "a__b", url: textUrl });
  assert.equal(badName.status, 400);
  assert.match(badName.body.error, /invalid upstream name/);
  const empty = await api("POST", "/servers", { name: "x" });
  assert.equal(empty.status, 400);
  const badMd = await api("POST", "/servers", { name: "y", markdown: "no tools" });
  assert.equal(badMd.status, 400);
});

test("a folder server never replaces one registered by env or API with the same name", async () => {
  fs.writeFileSync(path.join(dir, "text.md"), "## reverse\nFake reverse.\n- text (string, required): t\n");
  fs.writeFileSync(path.join(dir, "touch.md"), "## noop\nNothing.\n");
  await eventually(async () => (await toolNames()).includes("touch__noop"));
  const reversed = await rpc("tools/call", { name: "text__reverse", arguments: { text: "abc" } });
  assert.equal(reversed.content[0].text, "cba");
});
