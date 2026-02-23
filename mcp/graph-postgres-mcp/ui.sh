#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST="${UI_HOST:-127.0.0.1}"
PORT="${UI_PORT:-4000}"
URL="http://${HOST}:${PORT}"

if [ ! -f "$PROJECT_DIR/dist/resolvers.js" ] || [ ! -f "$PROJECT_DIR/dist/schema-generator.js" ] || [ ! -f "$PROJECT_DIR/dist/pg-client.js" ]; then
  echo "Build artifacts not found. Running npm run build..."
  cd "$PROJECT_DIR"
  npm run build
fi

if ! podman exec graphmcp-postgres pg_isready -U graphmcp -d graphmcpdb > /dev/null 2>&1; then
  echo "PostgreSQL container is not running. Run ./start.sh first."
  exit 1
fi

TMP_JS="$(mktemp /tmp/graphmcp-ui.XXXXXX.js)"

cat > "$TMP_JS" <<'EOF'
const http = require("http");
const { URL } = require("url");
const path = require("path");

const projectDir = process.env.GRAPHMCP_PROJECT_DIR;
const host = process.env.GRAPHMCP_UI_HOST || "127.0.0.1";
const port = Number(process.env.GRAPHMCP_UI_PORT || "4000");

const schemaModule = require(path.join(projectDir, "dist/schema-generator.js"));
const resolversModule = require(path.join(projectDir, "dist/resolvers.js"));
const pgModule = require(path.join(projectDir, "dist/pg-client.js"));

const { refreshSchema, getSchemaSDL, getTableSchemas } = schemaModule;
const { executeGraphQL } = resolversModule;
const { shutdown } = pgModule;

function send(res, status, body, contentType) {
  res.writeHead(status, {
    "Content-Type": contentType,
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function sendJson(res, status, data) {
  send(res, status, JSON.stringify(data, null, 2), "application/json; charset=utf-8");
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
      if (data.length > 1024 * 1024) {
        reject(new Error("Body too large"));
        req.destroy();
      }
    });
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });
}

function pageHtml() {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Graph Postgres MCP UI</title>
<style>
:root{--bg:#f2efe7;--panel:#fffdf8;--ink:#1f2328;--muted:#667085;--line:#d8d2c7;--accent:#0b6bcb;--accent2:#0c8f6e;--bad:#b42318}
*{box-sizing:border-box}
body{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:radial-gradient(circle at 10% 10%,#fff7d6 0,#f2efe7 35%,#e8efe9 100%);color:var(--ink)}
.wrap{display:grid;grid-template-columns:340px 1fr;min-height:100vh;gap:14px;padding:14px}
.panel{background:rgba(255,253,248,.88);border:1px solid var(--line);border-radius:14px;backdrop-filter:blur(6px);overflow:hidden}
.side{display:grid;grid-template-rows:auto auto 1fr 1fr}
.block{padding:12px;border-bottom:1px solid var(--line)}
.block:last-child{border-bottom:0}
.title{font-size:13px;color:var(--muted);margin:0 0 8px 0;text-transform:uppercase;letter-spacing:.08em}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{border:1px solid var(--line);background:#fff;padding:8px 10px;border-radius:10px;color:var(--ink);cursor:pointer;font:inherit}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
button.good{background:var(--accent2);color:#fff;border-color:var(--accent2)}
button:disabled{opacity:.6;cursor:not-allowed}
.status{font-size:12px;color:var(--muted)}
.status.bad{color:var(--bad)}
.tables{display:grid;gap:8px;max-height:100%;overflow:auto;padding-right:4px}
.table{border:1px solid var(--line);border-radius:10px;padding:8px;background:#fff}
.table h4{margin:0 0 6px 0;font-size:13px}
.table .meta{font-size:11px;color:var(--muted);margin-bottom:6px}
.table ul{margin:0;padding-left:16px}
.table li{font-size:12px;line-height:1.35}
.main{display:grid;grid-template-rows:auto 1fr 1fr;min-height:100%}
.toolbar{padding:12px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.toolbar .left{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;border-bottom:1px solid var(--line)}
.stack{display:grid;grid-template-rows:auto 1fr;min-height:260px}
label{font-size:12px;color:var(--muted);margin-bottom:6px;display:block}
textarea,pre{width:100%;height:100%;margin:0;border:1px solid var(--line);border-radius:10px;background:#fff;padding:10px;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre;overflow:auto}
textarea{resize:vertical;min-height:200px}
#result{background:#0d1117;color:#e6edf3;border-color:#30363d}
#schema{background:#fbfaf6}
.footer{padding:12px;display:grid;grid-template-rows:auto 1fr;min-height:260px}
@media (max-width: 980px){.wrap{grid-template-columns:1fr}.side{grid-template-rows:auto auto 280px 280px}.grid{grid-template-columns:1fr}.main{grid-template-rows:auto auto auto}}
</style>
</head>
<body>
<div class="wrap">
  <section class="panel side">
    <div class="block">
      <p class="title">Connection</p>
      <div class="row">
        <button id="refreshSchema" class="good">Refresh Schema</button>
        <button id="reloadView">Reload View</button>
      </div>
      <div id="status" class="status">Loading...</div>
    </div>
    <div class="block">
      <p class="title">Starter</p>
      <div class="row">
        <button id="qTables">list_tables</button>
        <button id="qUsers">users</button>
        <button id="qPosts">posts</button>
      </div>
    </div>
    <div class="block" style="min-height:220px">
      <p class="title">Tables</p>
      <div id="tables" class="tables"></div>
    </div>
    <div class="block" style="min-height:220px">
      <p class="title">Schema SDL</p>
      <pre id="schema"></pre>
    </div>
  </section>
  <section class="panel main">
    <div class="toolbar">
      <div class="left">
        <button id="run" class="primary">Run Query</button>
        <span class="status">POST /graphql</span>
      </div>
      <div class="status" id="queryStatus"></div>
    </div>
    <div class="grid">
      <div class="stack">
        <label for="query">Query</label>
        <textarea id="query">{ list_tables }</textarea>
      </div>
      <div class="stack">
        <label for="variables">Variables JSON</label>
        <textarea id="variables">{}</textarea>
      </div>
    </div>
    <div class="footer">
      <label for="result">Result</label>
      <pre id="result"></pre>
    </div>
  </section>
</div>
<script>
const el = (id) => document.getElementById(id);
const state = { tables: [], sdl: "" };

function setStatus(text, bad) {
  const n = el("status");
  n.textContent = text;
  n.className = bad ? "status bad" : "status";
}

function setQueryStatus(text, bad) {
  const n = el("queryStatus");
  n.textContent = text;
  n.className = bad ? "status bad" : "status";
}

function renderTables() {
  const host = el("tables");
  host.innerHTML = "";
  if (!Array.isArray(state.tables) || state.tables.length === 0) {
    host.textContent = "No tables found";
    return;
  }
  for (const t of state.tables) {
    const card = document.createElement("div");
    card.className = "table";
    const h = document.createElement("h4");
    h.textContent = t.name;
    const m = document.createElement("div");
    m.className = "meta";
    m.textContent = "pk: " + ((t.primaryKeys || []).join(", ") || "none");
    const ul = document.createElement("ul");
    for (const c of t.columns || []) {
      const li = document.createElement("li");
      li.textContent = c.column_name + " : " + c.data_type + (c.is_nullable === "YES" ? "" : " !");
      ul.appendChild(li);
    }
    card.appendChild(h);
    card.appendChild(m);
    card.appendChild(ul);
    host.appendChild(card);
  }
}

async function loadView() {
  setStatus("Loading...", false);
  try {
    const [tablesRes, schemaRes] = await Promise.all([fetch("/tables"), fetch("/schema")]);
    const tablesData = await tablesRes.json();
    const schemaData = await schemaRes.json();
    state.tables = tablesData.tables || [];
    state.sdl = schemaData.sdl || "";
    renderTables();
    el("schema").textContent = state.sdl;
    setStatus("Loaded " + state.tables.length + " tables", false);
  } catch (e) {
    setStatus(String(e && e.message ? e.message : e), true);
  }
}

async function runQuery() {
  setQueryStatus("Running...", false);
  try {
    let variables = {};
    const rawVariables = el("variables").value.trim();
    if (rawVariables) {
      variables = JSON.parse(rawVariables);
    }
    const res = await fetch("/graphql", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: el("query").value, variables })
    });
    const data = await res.json();
    el("result").textContent = JSON.stringify(data, null, 2);
    setQueryStatus(res.ok ? "Done" : "Request failed", !res.ok);
  } catch (e) {
    el("result").textContent = JSON.stringify({ error: String(e && e.message ? e.message : e) }, null, 2);
    setQueryStatus("Error", true);
  }
}

async function refreshSchema() {
  setStatus("Refreshing...", false);
  try {
    const res = await fetch("/refresh", { method: "POST" });
    const data = await res.json();
    await loadView();
    setStatus("Refreshed " + data.tableCount + " tables", false);
  } catch (e) {
    setStatus(String(e && e.message ? e.message : e), true);
  }
}

el("run").addEventListener("click", runQuery);
el("refreshSchema").addEventListener("click", refreshSchema);
el("reloadView").addEventListener("click", loadView);
el("qTables").addEventListener("click", () => { el("query").value = "{ list_tables }"; });
el("qUsers").addEventListener("click", () => { el("query").value = "{ users(limit: 10) { id name email created_at } }"; });
el("qPosts").addEventListener("click", () => { el("query").value = "{ posts(limit: 10) { id user_id title published created_at } }"; });
loadView();
</script>
</body>
</html>`;
}

const server = http.createServer(async (req, res) => {
  try {
    if (!req.url) {
      sendJson(res, 400, { error: "Missing URL" });
      return;
    }
    const url = new URL(req.url, `http://${host}:${port}`);
    if (req.method === "GET" && url.pathname === "/") {
      send(res, 200, pageHtml(), "text/html; charset=utf-8");
      return;
    }
    if (req.method === "GET" && url.pathname === "/schema") {
      sendJson(res, 200, { sdl: getSchemaSDL() });
      return;
    }
    if (req.method === "GET" && url.pathname === "/tables") {
      sendJson(res, 200, { tables: getTableSchemas() });
      return;
    }
    if (req.method === "POST" && url.pathname === "/refresh") {
      const out = await refreshSchema();
      sendJson(res, 200, out);
      return;
    }
    if ((req.method === "POST" || req.method === "GET") && url.pathname === "/graphql") {
      let query = "";
      let variables;
      if (req.method === "GET") {
        query = url.searchParams.get("query") || "";
        const rawVars = url.searchParams.get("variables");
        if (rawVars) {
          variables = JSON.parse(rawVars);
        }
      } else {
        const raw = await readBody(req);
        const body = raw ? JSON.parse(raw) : {};
        query = typeof body.query === "string" ? body.query : "";
        variables = body.variables;
      }
      if (!query) {
        sendJson(res, 400, { error: "query is required" });
        return;
      }
      const result = await executeGraphQL(query, variables);
      sendJson(res, 200, result);
      return;
    }
    sendJson(res, 404, { error: "Not found" });
  } catch (err) {
    sendJson(res, 500, { error: err && err.message ? err.message : String(err) });
  }
});

async function start() {
  await refreshSchema();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, resolve);
  });
  process.stdout.write(`Graph UI running at http://${host}:${port}\n`);
}

async function stop(code) {
  try {
    await new Promise((resolve) => server.close(() => resolve()));
  } catch (_) {
  }
  try {
    await shutdown();
  } catch (_) {
  }
  process.exit(code);
}

process.on("SIGINT", () => { stop(0); });
process.on("SIGTERM", () => { stop(0); });

start().catch((err) => {
  process.stderr.write(`Failed to start UI: ${err && err.message ? err.message : String(err)}\n`);
  stop(1);
});
EOF

cleanup() {
  rm -f "$TMP_JS"
}

trap cleanup EXIT

GRAPHMCP_PROJECT_DIR="$PROJECT_DIR" GRAPHMCP_UI_HOST="$HOST" GRAPHMCP_UI_PORT="$PORT" node "$TMP_JS" &
SERVER_PID=$!

sleep 1

if [ "${UI_NO_OPEN:-0}" != "1" ]; then
  if command -v open > /dev/null 2>&1; then
    open "$URL" > /dev/null 2>&1 || true
  elif command -v xdg-open > /dev/null 2>&1; then
    xdg-open "$URL" > /dev/null 2>&1 || true
  fi
fi

echo "UI available at $URL"
wait "$SERVER_PID"
