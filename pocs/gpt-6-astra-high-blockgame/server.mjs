import http from "node:http";
import { globSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(fileURLToPath(import.meta.url));
const ports = await readFile(path.join(root, "scripts/ports.env"), "utf8");
const port = Number(ports.match(/^WEB=(\d+)$/m)[1]);
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".woff2": "font/woff2",
};
const server = http.createServer(async (req, res) => {
  const pathname = new URL(req.url, "http://localhost").pathname;
  if (pathname === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    return res.end(JSON.stringify({ status: "ok", app: "brisa" }));
  }
  const allowed =
    pathname === "/" ||
    pathname === "/index.html" ||
    /^\/assets\/[a-zA-Z0-9_.-]+$/.test(pathname) ||
    [
      "/node_modules/three/build/three.module.js",
      "/node_modules/three/build/three.core.js",
    ].includes(pathname);
  if (!allowed) {
    res.writeHead(404);
    return res.end("Not found");
  }
  try {
    const relative =
      pathname === "/assets/OrbitControls.js"
        ? globSync("node_modules/three/*/jsm/controls/OrbitControls.js", {
            cwd: root,
          })[0]
        : pathname === "/"
          ? "index.html"
          : pathname;
    const file = path.join(root, relative);
    const body = await readFile(file);
    res.writeHead(200, {
      "Content-Type": types[path.extname(file)] || "application/octet-stream",
      "Cache-Control": "no-cache",
      "X-Content-Type-Options": "nosniff",
    });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
});
server.listen(port, "0.0.0.0", () =>
  process.stdout.write(`Brisa is running at http://localhost:${port}\n`),
);
