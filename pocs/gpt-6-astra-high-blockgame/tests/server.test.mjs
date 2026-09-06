import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawn } from "node:child_process";
const port = Number(
  readFileSync(new URL("../scripts/ports.env", import.meta.url), "utf8").match(
    /^WEB=(\d+)$/m,
  )[1],
);
const base = `http://localhost:${port}`;
let child;
test("The server serves the game and blocks access to project internals", async (t) => {
  try {
    const res = await fetch(`${base}/health`);
    assert.equal((await res.json()).app, "brisa");
  } catch {
    child = spawn(process.execPath, ["server.mjs"], {
      cwd: new URL("..", import.meta.url),
      stdio: ["ignore", "pipe", "pipe"],
    });
    await new Promise((resolve, reject) => {
      child.stdout.once("data", resolve);
      child.once("error", reject);
      child.once("exit", (code) => reject(new Error(`Server exited ${code}`)));
    });
  }
  t.after(() => child?.kill());
  const health = await fetch(`${base}/health`);
  assert.deepEqual(await health.json(), { status: "ok", app: "brisa" });
  const index = await fetch(base);
  assert.equal(index.status, 200);
  assert.match(await index.text(), /Costa do Sol/);
  for (const path of [
    "/package.json",
    "/.git/config",
    "/server.mjs",
    "/scripts/ports.env",
    "/assets/..%2fserver.mjs",
  ])
    assert.equal((await fetch(base + path)).status, 404, path);
  assert.match(
    (
      await fetch(`${base}/node_modules/three/build/three.module.js`)
    ).headers.get("content-type"),
    /javascript/,
  );
});
