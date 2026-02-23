#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST="${UI_HOST:-127.0.0.1}"
PORT="${UI_PORT:-4000}"
URL="http://${HOST}:${PORT}"
PG_CONTAINER="${PG_CONTAINER_NAME:-graphmcp-postgres}"
PG_USER_VALUE="${PG_USER:-graphmcp}"
PG_DATABASE_VALUE="${PG_DATABASE:-graphmcpdb}"

if [ ! -f "$PROJECT_DIR/dist/index.js" ]; then
  echo "Build artifacts not found. Running npm run build..."
  cd "$PROJECT_DIR"
  npm run build
fi

if [ "${UI_SKIP_PG_CHECK:-0}" != "1" ] && command -v podman > /dev/null 2>&1; then
  if ! podman exec "$PG_CONTAINER" pg_isready -U "$PG_USER_VALUE" -d "$PG_DATABASE_VALUE" > /dev/null 2>&1; then
    echo "PostgreSQL container is not ready. Run ./start.sh first or set UI_SKIP_PG_CHECK=1."
    exit 1
  fi
fi

TMP_JS="$(mktemp /tmp/graphmcp-ui.XXXXXX.js)"

cat > "$TMP_JS" <<'EOF'
const http = require("http");
const { URL } = require("url");
const path = require("path");
const { createRequire } = require("module");

const projectDir = process.env.GRAPHMCP_PROJECT_DIR;
const host = process.env.GRAPHMCP_UI_HOST || "127.0.0.1";
const port = Number(process.env.GRAPHMCP_UI_PORT || "4000");
const projectRequire = createRequire(path.join(projectDir, "package.json"));
const { Client } = projectRequire("@modelcontextprotocol/sdk/client");
const { StdioClientTransport } = projectRequire("@modelcontextprotocol/sdk/client/stdio.js");

const mcpEnv = {};
for (const k of ["PG_HOST", "PG_PORT", "PG_USER", "PG_PASSWORD", "PG_DATABASE"]) {
  if (typeof process.env[k] === "string" && process.env[k] !== "") {
    mcpEnv[k] = process.env[k];
  }
}

let client = null;
let transport = null;
let mcpConnected = false;
let mcpTools = [];
let mcpLastError = null;

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

function getTextContent(result) {
  const items = Array.isArray(result && result.content) ? result.content : [];
  return items.filter((x) => x && x.type === "text").map((x) => x.text || "").join("\n");
}

async function connectMcp() {
  if (mcpConnected && client && transport) {
    return;
  }
  client = new Client({ name: "graph-postgres-ui", version: "1.0.0" });
  transport = new StdioClientTransport({
    command: "node",
    args: [path.join(projectDir, "dist/index.js")],
    env: Object.keys(mcpEnv).length > 0 ? mcpEnv : undefined,
    cwd: projectDir,
    stderr: "pipe",
  });
  if (transport.stderr) {
    transport.stderr.on("data", (chunk) => {
      const line = String(chunk || "").trim();
      if (line) {
        mcpLastError = line;
      }
    });
  }
  client.onerror = (err) => {
    mcpLastError = err && err.message ? err.message : String(err);
  };
  await client.connect(transport);
  mcpConnected = true;
  const tools = await client.listTools();
  mcpTools = tools.tools || [];
}

async function callTool(name, args) {
  await connectMcp();
  const result = await client.callTool({ name, arguments: args || {} });
  return result;
}

async function callToolJson(name, args) {
  const result = await callTool(name, args);
  const text = getTextContent(result);
  if (!text) {
    return {};
  }
  return JSON.parse(text);
}

function pageHtml() {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Graph Postgres MCP UI</title>
<style>
:root{--bg:#efece2;--ink:#1a1f24;--muted:#6a737d;--line:#d7d1c5;--panel:#fffcf6ee;--accent:#0d6efd;--good:#0f8f5f;--bad:#b42318;--left-w:360px;--left-top-h:52%;--right-top-h:56%;--rt-left-w:52%;--rb-left-w:38%}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:radial-gradient(circle at 15% 15%,#fff5cc 0,#efece2 45%,#e4eee7 100%);color:var(--ink)}
.app{height:100vh;display:grid;grid-template-columns:minmax(260px,var(--left-w)) 10px minmax(500px,1fr);gap:0;padding:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;min-width:0;min-height:0;backdrop-filter:blur(6px)}
.split{background:transparent;position:relative}
.split::before{content:"";position:absolute;inset:1px;border-radius:8px;background:linear-gradient(180deg,#d3ccbf,#c7c0b4)}
.split.v{cursor:col-resize}
.split.h{cursor:row-resize}
.left{display:grid;grid-template-rows:minmax(180px,var(--left-top-h)) 10px minmax(140px,1fr)}
.pane{display:grid;grid-template-rows:auto 1fr;min-height:0;min-width:0}
.paneHead{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px 12px;border-bottom:1px solid var(--line);background:#fffdf9}
.paneTitle{margin:0;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.paneBody{padding:10px;overflow:auto;min-height:0}
.right{display:grid;grid-template-rows:minmax(260px,var(--right-top-h)) 10px minmax(220px,1fr)}
.grid2{display:grid;min-height:0;min-width:0}
.topGrid{grid-template-columns:minmax(220px,var(--rt-left-w)) 10px minmax(220px,1fr)}
.bottomGrid{grid-template-columns:minmax(220px,var(--rb-left-w)) 10px minmax(220px,1fr)}
.stack{display:grid;grid-template-rows:auto auto 1fr;min-height:0}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 12px;border-bottom:1px solid var(--line);background:#fffdf9}
.controls.slim{padding:8px 10px}
.status{font-size:12px;color:var(--muted)}
.status.bad{color:var(--bad)}
button{font:inherit;border:1px solid var(--line);background:#fff;padding:8px 10px;border-radius:10px;color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.good{background:var(--good);border-color:var(--good);color:#fff}
textarea,pre{width:100%;height:100%;margin:0;border:1px solid var(--line);border-radius:10px;padding:10px;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre;overflow:auto;background:#fff}
textarea{resize:none;min-height:0}
#result{background:#0d1117;color:#e6edf3;border-color:#30363d}
#schema{background:#fbfaf6}
.tables{display:grid;gap:8px}
.card{border:1px solid var(--line);border-radius:10px;background:#fff;padding:8px}
.card h4{margin:0 0 6px 0;font-size:13px}
.meta{font-size:11px;color:var(--muted);margin-bottom:6px}
.card ul{margin:0;padding-left:16px}
.card li{font-size:12px;line-height:1.35}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{font-size:11px;border:1px solid var(--line);background:#fff;border-radius:999px;padding:4px 8px}
.toolList{display:grid;gap:8px}
.toolCard{border:1px solid var(--line);border-radius:10px;background:#fff;padding:8px}
.toolCard b{font-size:12px}
.toolCard div{font-size:11px;color:var(--muted);margin-top:4px;line-height:1.35}
.fill{display:grid;min-height:0}
.label{font-size:12px;color:var(--muted);padding:8px 10px 0}
@media (max-width:1100px){.app{grid-template-columns:1fr;grid-template-rows:minmax(260px,42vh) 10px minmax(420px,1fr)}#split-main{cursor:row-resize}.app.vertical{grid-template-columns:1fr;grid-template-rows:minmax(260px,var(--left-w)) 10px minmax(500px,1fr)}}
</style>
</head>
<body>
<div class="app" id="appRoot">
  <section class="panel left" id="leftPanel">
    <div class="pane">
      <div class="paneHead">
        <p class="paneTitle">Explorer</p>
        <div class="chips">
          <span class="chip" id="dbStatus">Connecting</span>
          <span class="chip" id="toolCount">0 tools</span>
        </div>
      </div>
      <div class="paneBody">
        <div class="controls slim">
          <button id="refreshSchema" class="good">Refresh Schema</button>
          <button id="reloadView">Reload View</button>
          <button id="qTables">list_tables</button>
          <span id="starterButtons"></span>
        </div>
        <div class="status" id="status" style="padding:10px 2px">Loading...</div>
        <div id="tables" class="tables"></div>
      </div>
    </div>
    <div class="split h" id="split-left"></div>
    <div class="pane">
      <div class="paneHead">
        <p class="paneTitle">MCP Tools</p>
      </div>
      <div class="paneBody">
        <div id="tools" class="toolList"></div>
      </div>
    </div>
  </section>
  <div class="split v" id="split-main"></div>
  <section class="panel right" id="rightPanel">
    <div class="grid2 topGrid" id="topGrid">
      <section class="pane stack">
        <div class="paneHead">
          <p class="paneTitle">GraphQL Query</p>
        </div>
        <div class="controls">
          <button id="run" class="primary">Run Query</button>
          <span class="status">MCP tool: graphql_query</span>
          <span class="status" id="queryStatus"></span>
        </div>
        <div class="paneBody fill">
          <textarea id="query">{ list_tables }</textarea>
        </div>
      </section>
      <div class="split v" id="split-rt"></div>
      <section class="pane stack">
        <div class="paneHead">
          <p class="paneTitle">Result</p>
        </div>
        <div class="label">Top-right result panel</div>
        <div class="paneBody fill">
          <pre id="result"></pre>
        </div>
      </section>
    </div>
    <div class="split h" id="split-right"></div>
    <div class="grid2 bottomGrid" id="bottomGrid">
      <section class="pane stack">
        <div class="paneHead">
          <p class="paneTitle">Variables JSON</p>
        </div>
        <div class="label">Sent to graphql_query.variables</div>
        <div class="paneBody fill">
          <textarea id="variables">{}</textarea>
        </div>
      </section>
      <div class="split v" id="split-rb"></div>
      <section class="pane stack">
        <div class="paneHead">
          <p class="paneTitle">Schema SDL</p>
        </div>
        <div class="label">Bottom-right schema panel</div>
        <div class="paneBody fill">
          <pre id="schema"></pre>
        </div>
      </section>
    </div>
  </section>
</div>
<script>
const el = (id) => document.getElementById(id)
const state = { tables: [], sdl: "", tools: [], health: null }

function setStatus(text, bad) {
  const n = el("status")
  n.textContent = text
  n.className = bad ? "status bad" : "status"
}

function setQueryStatus(text, bad) {
  const n = el("queryStatus")
  n.textContent = text
  n.className = bad ? "status bad" : "status"
}

function renderTables() {
  const host = el("tables")
  host.innerHTML = ""
  if (!Array.isArray(state.tables) || state.tables.length === 0) {
    host.textContent = "No tables found"
    return
  }
  for (const t of state.tables) {
    const card = document.createElement("div")
    card.className = "card"
    const h = document.createElement("h4")
    h.textContent = t.table || t.name || ""
    const m = document.createElement("div")
    m.className = "meta"
    const pk = Array.isArray(t.primaryKeys) ? t.primaryKeys.join(", ") : ""
    m.textContent = "pk: " + (pk || "none")
    const ul = document.createElement("ul")
    const cols = Array.isArray(t.columns) ? t.columns : []
    for (const c of cols) {
      const li = document.createElement("li")
      const cname = c.name || c.column_name || ""
      const ctype = c.type || c.data_type || "unknown"
      const nullable = c.nullable === true || c.is_nullable === "YES"
      li.textContent = cname + " : " + ctype + (nullable ? "" : " !")
      ul.appendChild(li)
    }
    card.appendChild(h)
    card.appendChild(m)
    card.appendChild(ul)
    host.appendChild(card)
  }
}

function renderStarterButtons() {
  const host = el("starterButtons")
  host.innerHTML = ""
  const tables = Array.isArray(state.tables) ? state.tables : []
  for (const t of tables) {
    const tableName = t.table || t.name || ""
    if (!tableName) continue
    const cols = (Array.isArray(t.columns) ? t.columns : [])
      .map((c) => c.name || c.column_name || "")
      .filter(Boolean)
      .slice(0, 6)
    const fields = cols.length > 0 ? cols.join(" ") : "id"
    const btn = document.createElement("button")
    btn.textContent = tableName
    btn.addEventListener("click", () => {
      el("query").value = "{ " + tableName + "(limit: 10) { " + fields + " } }"
    })
    host.appendChild(btn)
  }
}

function renderTools() {
  const host = el("tools")
  host.innerHTML = ""
  const tools = Array.isArray(state.tools) ? state.tools : []
  el("toolCount").textContent = tools.length + " tools"
  if (tools.length === 0) {
    host.textContent = "No tools loaded"
    return
  }
  for (const t of tools) {
    const card = document.createElement("div")
    card.className = "toolCard"
    const name = document.createElement("b")
    name.textContent = t.name
    const desc = document.createElement("div")
    desc.textContent = (t.description || "").slice(0, 220)
    card.appendChild(name)
    card.appendChild(desc)
    host.appendChild(card)
  }
}

async function loadHealth() {
  const res = await fetch("/health")
  const data = await res.json()
  state.health = data
  el("dbStatus").textContent = data.connected ? "MCP connected" : "MCP error"
}

async function loadTools() {
  const res = await fetch("/tools")
  const data = await res.json()
  state.tools = data.tools || []
  renderTools()
}

async function loadView() {
  setStatus("Loading...", false)
  try {
    const [tablesRes, schemaRes] = await Promise.all([fetch("/tables"), fetch("/schema")])
    const tablesData = await tablesRes.json()
    const schemaData = await schemaRes.json()
    state.tables = tablesData.tables || []
    state.sdl = schemaData.sdl || ""
    renderTables()
    renderStarterButtons()
    el("schema").textContent = state.sdl
    await Promise.all([loadHealth(), loadTools()])
    setStatus("Loaded " + state.tables.length + " tables via MCP", false)
  } catch (e) {
    setStatus(String(e && e.message ? e.message : e), true)
  }
}

async function runQuery() {
  setQueryStatus("Running...", false)
  try {
    let variables = {}
    const rawVariables = el("variables").value.trim()
    if (rawVariables) {
      variables = JSON.parse(rawVariables)
    }
    const res = await fetch("/graphql", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: el("query").value, variables })
    })
    const data = await res.json()
    el("result").textContent = JSON.stringify(data, null, 2)
    setQueryStatus(res.ok ? "Done" : "Failed", !res.ok)
  } catch (e) {
    el("result").textContent = JSON.stringify({ error: String(e && e.message ? e.message : e) }, null, 2)
    setQueryStatus("Error", true)
  }
}

async function refreshSchema() {
  setStatus("Refreshing...", false)
  try {
    const res = await fetch("/refresh", { method: "POST" })
    const data = await res.json()
    await loadView()
    setStatus("Refreshed " + data.tableCount + " tables", false)
  } catch (e) {
    setStatus(String(e && e.message ? e.message : e), true)
  }
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n))
}

function setupSplitters() {
  const app = el("appRoot")
  const mobile = () => window.matchMedia("(max-width:1100px)").matches
  function bind(id, axis, getRect, apply) {
    const s = el(id)
    let active = false
    let start = 0
    let startValue = 0
    s.addEventListener("pointerdown", (ev) => {
      active = true
      s.setPointerCapture(ev.pointerId)
      const rect = getRect()
      start = axis === "x" ? ev.clientX : ev.clientY
      startValue = axis === "x" ? rect.valueX : rect.valueY
    })
    s.addEventListener("pointermove", (ev) => {
      if (!active) return
      const rect = getRect()
      const pos = axis === "x" ? ev.clientX : ev.clientY
      const delta = pos - start
      const next = startValue + delta
      apply(next, rect)
    })
    s.addEventListener("pointerup", (ev) => {
      active = false
      try { s.releasePointerCapture(ev.pointerId) } catch (_) {}
    })
    s.addEventListener("pointercancel", () => { active = false })
  }

  bind("split-main", "x", () => {
    if (mobile()) {
      const r = app.getBoundingClientRect()
      const current = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--left-w")) || r.height * 0.42
      return { r, valueY: current }
    }
    const r = app.getBoundingClientRect()
    const current = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--left-w")) || 360
    return { r, valueX: current }
  }, (next, info) => {
    if (mobile()) {
      const h = info.r.height
      document.documentElement.style.setProperty("--left-w", clamp(next, 220, h - 260) + "px")
      return
    }
    const w = info.r.width
    document.documentElement.style.setProperty("--left-w", clamp(next, 260, w - 420) + "px")
  })

  bind("split-left", "y", () => {
    const r = el("leftPanel").getBoundingClientRect()
    const current = (parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--left-top-h")) || 52)
    return { r, valueY: (current / 100) * r.height }
  }, (next, info) => {
    const pct = clamp((next / info.r.height) * 100, 28, 75)
    document.documentElement.style.setProperty("--left-top-h", pct + "%")
  })

  bind("split-right", "y", () => {
    const r = el("rightPanel").getBoundingClientRect()
    const current = (parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--right-top-h")) || 56)
    return { r, valueY: (current / 100) * r.height }
  }, (next, info) => {
    const pct = clamp((next / info.r.height) * 100, 30, 75)
    document.documentElement.style.setProperty("--right-top-h", pct + "%")
  })

  bind("split-rt", "x", () => {
    const r = el("topGrid").getBoundingClientRect()
    const current = (parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--rt-left-w")) || 52)
    return { r, valueX: (current / 100) * r.width }
  }, (next, info) => {
    const pct = clamp((next / info.r.width) * 100, 28, 72)
    document.documentElement.style.setProperty("--rt-left-w", pct + "%")
  })

  bind("split-rb", "x", () => {
    const r = el("bottomGrid").getBoundingClientRect()
    const current = (parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--rb-left-w")) || 38)
    return { r, valueX: (current / 100) * r.width }
  }, (next, info) => {
    const pct = clamp((next / info.r.width) * 100, 24, 72)
    document.documentElement.style.setProperty("--rb-left-w", pct + "%")
  })
}

el("run").addEventListener("click", runQuery)
el("refreshSchema").addEventListener("click", refreshSchema)
el("reloadView").addEventListener("click", loadView)
el("qTables").addEventListener("click", () => { el("query").value = "{ list_tables }" })
el("query").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault()
    runQuery()
  }
})

setupSplitters()
loadView()
</script>
</body>
</html>`
}

const server = http.createServer(async (req, res) => {
  try {
    if (!req.url) {
      sendJson(res, 400, { error: "Missing URL" })
      return
    }
    const url = new URL(req.url, `http://${host}:${port}`)
    if (req.method === "GET" && url.pathname === "/") {
      send(res, 200, pageHtml(), "text/html; charset=utf-8")
      return
    }
    if (req.method === "GET" && url.pathname === "/health") {
      try {
        await connectMcp()
        sendJson(res, 200, { connected: true, tools: mcpTools.map((t) => t.name), lastError: mcpLastError })
      } catch (err) {
        mcpLastError = err && err.message ? err.message : String(err)
        sendJson(res, 500, { connected: false, lastError: mcpLastError })
      }
      return
    }
    if (req.method === "GET" && url.pathname === "/tools") {
      await connectMcp()
      const tools = await client.listTools()
      mcpTools = tools.tools || []
      sendJson(res, 200, { tools: mcpTools })
      return
    }
    if (req.method === "GET" && url.pathname === "/schema") {
      const result = await callTool("get_schema", {})
      sendJson(res, 200, { sdl: getTextContent(result) })
      return
    }
    if (req.method === "GET" && url.pathname === "/tables") {
      const result = await callToolJson("list_tables", {})
      sendJson(res, 200, { tables: result })
      return
    }
    if (req.method === "POST" && url.pathname === "/refresh") {
      const raw = await callTool("refresh_schema", {})
      const text = getTextContent(raw)
      let tableCount = null
      let fieldCount = null
      const match = /Schema refreshed: (\\d+) tables, (\\d+) fields/.exec(text)
      if (match) {
        tableCount = Number(match[1])
        fieldCount = Number(match[2])
      }
      sendJson(res, 200, { message: text, tableCount, fieldCount })
      return
    }
    if ((req.method === "POST" || req.method === "GET") && url.pathname === "/graphql") {
      let query = ""
      let variables
      if (req.method === "GET") {
        query = url.searchParams.get("query") || ""
        const rawVars = url.searchParams.get("variables")
        if (rawVars) {
          variables = JSON.parse(rawVars)
        }
      } else {
        const raw = await readBody(req)
        const body = raw ? JSON.parse(raw) : {}
        query = typeof body.query === "string" ? body.query : ""
        variables = body.variables
      }
      if (!query) {
        sendJson(res, 400, { error: "query is required" })
        return
      }
      const args = { query }
      if (variables !== undefined) {
        args.variables = JSON.stringify(variables)
      }
      const out = await callToolJson("graphql_query", args)
      sendJson(res, 200, out)
      return
    }
    sendJson(res, 404, { error: "Not found" })
  } catch (err) {
    mcpLastError = err && err.message ? err.message : String(err)
    sendJson(res, 500, { error: mcpLastError })
  }
})

async function start() {
  await connectMcp()
  await new Promise((resolve, reject) => {
    server.once("error", reject)
    server.listen(port, host, resolve)
  })
  process.stdout.write(`Graph UI running at http://${host}:${port}\n`)
}

async function closeMcp() {
  try {
    if (transport) {
      await transport.close()
    }
  } catch (_) {
  }
  transport = null
  client = null
  mcpConnected = false
}

async function stop(code) {
  try {
    await new Promise((resolve) => server.close(() => resolve()))
  } catch (_) {
  }
  await closeMcp()
  process.exit(code)
}

process.on("SIGINT", () => { stop(0) })
process.on("SIGTERM", () => { stop(0) })

start().catch((err) => {
  process.stderr.write(`Failed to start UI: ${err && err.message ? err.message : String(err)}\n`)
  stop(1)
})
EOF

cleanup() {
  rm -f "$TMP_JS"
}

trap cleanup EXIT

GRAPHMCP_PROJECT_DIR="$PROJECT_DIR" GRAPHMCP_UI_HOST="$HOST" GRAPHMCP_UI_PORT="$PORT" \
PG_HOST="${PG_HOST:-localhost}" PG_PORT="${PG_PORT:-5432}" PG_USER="${PG_USER:-graphmcp}" PG_PASSWORD="${PG_PASSWORD:-graphmcp123}" PG_DATABASE="${PG_DATABASE:-graphmcpdb}" \
node "$TMP_JS" &
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
