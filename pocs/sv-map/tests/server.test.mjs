import test from "node:test";
import assert from "node:assert/strict";
import { createServer, loadCompanies } from "../server/server.mjs";

const server = createServer();
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const base = `http://127.0.0.1:${server.address().port}`;
const get = path => fetch(base + path);
test.after(() => server.close());

test("health reports every company so the boot screen can confirm the data loaded", async () => {
  const body = await (await get("/api/health")).json();
  assert.deepEqual(body, { status: "ok", companies: loadCompanies().length });
});

test("search by name returns that company first with its address", async () => {
  const [first] = await (await get("/api/search?q=nvidia")).json();
  assert.equal(first.id, "nvidia");
  assert.match(first.address, /\d/);
});

test("search honors the category filter", async () => {
  const hits = await (await get("/api/search?q=&category=ailab")).json();
  assert.ok(hits.length > 0);
  assert.ok(hits.every(c => c.category === "ailab"));
});

test("the API flags which companies have a logo so the UI draws initials instead of requesting a missing file", async () => {
  const companies = await (await get("/api/companies")).json();
  assert.equal(companies.find(c => c.id === "google").logo, true);
  assert.equal(companies.find(c => c.id === "ssi").logo, false);
});

test("a single company can be fetched by id and unknown ids are 404", async () => {
  assert.equal((await (await get("/api/companies/openai")).json()).name, "OpenAI");
  assert.equal((await get("/api/companies/nope")).status, 404);
});

test("the UI, leaflet and logos are served but files outside the mounts are not", async () => {
  assert.match(await (await get("/")).text(), /leaflet\.js/);
  assert.equal((await get("/vendor/leaflet/leaflet.js")).status, 200);
  assert.equal((await get("/logos/google.png")).headers.get("content-type"), "image/png");
  assert.equal((await get("/..%2Fpackage.json")).status, 404);
  assert.equal((await get("/vendor/leaflet/..%2F..%2F..%2Fpackage.json")).status, 404);
});
