import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { searchCompanies } from "../web/search.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MOUNTS = [
  ["/vendor/leaflet/", path.join(ROOT, "node_modules/leaflet/dist")],
  ["/logos/", path.join(ROOT, "data/logos")],
  ["/", path.join(ROOT, "web")]
];
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml"
};

export function loadCompanies() {
  return JSON.parse(fs.readFileSync(path.join(ROOT, "data/companies.json"), "utf8"))
    .map(company => ({ ...company, logo: fs.existsSync(path.join(ROOT, "data/logos", `${company.id}.png`)) }));
}

function sendJson(res, status, body) {
  res.writeHead(status, { "Content-Type": TYPES[".json"], "Cache-Control": "no-store" });
  res.end(JSON.stringify(body));
}

function resolveStatic(urlPath) {
  for (const [prefix, dir] of MOUNTS) {
    if (!urlPath.startsWith(prefix)) continue;
    const relative = urlPath.slice(prefix.length) || "index.html";
    const file = path.resolve(dir, relative);
    if (!file.startsWith(dir + path.sep)) return null;
    return file;
  }
  return null;
}

function serveStatic(res, urlPath) {
  const file = resolveStatic(decodeURIComponent(urlPath));
  if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    return sendJson(res, 404, { error: "not found" });
  }
  res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
}

export function createServer() {
  const companies = loadCompanies();
  return http.createServer((req, res) => {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname === "/api/health") return sendJson(res, 200, { status: "ok", companies: companies.length });
    if (url.pathname === "/api/companies") return sendJson(res, 200, companies);
    if (url.pathname === "/api/search") {
      const hits = searchCompanies(companies, url.searchParams.get("q"), url.searchParams.get("category") || "all");
      return sendJson(res, 200, hits);
    }
    const id = url.pathname.match(/^\/api\/companies\/([a-z0-9-]+)$/)?.[1];
    if (id) {
      const company = companies.find(c => c.id === id);
      return company ? sendJson(res, 200, company) : sendJson(res, 404, { error: "not found" });
    }
    serveStatic(res, url.pathname);
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 8097);
  createServer().listen(port, "127.0.0.1", () => console.log(`sv-map server on http://localhost:${port}`));
}
