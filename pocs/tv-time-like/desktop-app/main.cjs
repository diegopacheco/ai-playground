const { app, BrowserWindow, dialog, shell } = require("electron")
const { execFileSync, spawn } = require("node:child_process")
const fs = require("node:fs")
const http = require("node:http")
const os = require("node:os")
const path = require("node:path")

const sourceRoot = path.resolve(__dirname, "..")
const runtimeRoot = app.isPackaged ? app.getAppPath() : sourceRoot
const distRoot = app.isPackaged ? path.join(runtimeRoot, "web") : path.join(sourceRoot, "dist")
const apiOrigin = "http://127.0.0.1:3001"
let backend
let webServer
let mainWindow

const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".webp": "image/webp"
}

const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds))

const databasePath = () => {
  if (!app.isPackaged) return path.join(runtimeRoot, "data", "reelmark.db")
  const config = path.join(runtimeRoot, "config.plist")
  return execFileSync("/usr/bin/plutil", ["-extract", "DatabasePath", "raw", config], { encoding: "utf8" }).trim()
}

const apiReady = async () => {
  try {
    const response = await fetch(`${apiOrigin}/api/health`)
    return response.ok
  } catch {
    return false
  }
}

const startBackend = async () => {
  if (await apiReady()) return
  const bunCandidates = [
    process.env.REELMARK_BUN,
    path.join(os.homedir(), ".bun", "bin", "bun"),
    "/opt/homebrew/bin/bun",
    "/usr/local/bin/bun"
  ].filter(Boolean)
  const bun = bunCandidates.find(candidate => fs.existsSync(candidate)) || "bun"
  const args = app.isPackaged ? ["server/index.ts"] : ["run", "server"]
  const pathEntries = [path.join(os.homedir(), ".bun", "bin"), "/opt/homebrew/bin", "/usr/local/bin", process.env.PATH].filter(Boolean)
  backend = spawn(bun, args, {
    cwd: runtimeRoot,
    env: { ...process.env, DB_PATH: databasePath(), PATH: pathEntries.join(":") },
    stdio: "ignore"
  })
  for (let attempt = 0; attempt < 120; attempt += 1) {
    if (await apiReady()) return
    if (backend.exitCode !== null) break
    await delay(500)
  }
  throw new Error("The Reelmark API could not start.")
}

const proxyAPI = async (request, response) => {
  const chunks = []
  for await (const chunk of request) chunks.push(chunk)
  const headers = { ...request.headers }
  delete headers.host
  delete headers["content-length"]
  const method = request.method || "GET"
  const upstream = await fetch(`${apiOrigin}${request.url}`, {
    method,
    headers,
    body: method === "GET" || method === "HEAD" ? undefined : Buffer.concat(chunks)
  })
  response.statusCode = upstream.status
  upstream.headers.forEach((value, key) => {
    if (!["connection", "content-encoding", "transfer-encoding"].includes(key)) response.setHeader(key, value)
  })
  response.end(Buffer.from(await upstream.arrayBuffer()))
}

const resolveAsset = requestURL => {
  const pathname = decodeURIComponent(new URL(requestURL, "http://localhost").pathname)
  const relative = pathname === "/" ? "index.html" : pathname.slice(1)
  const candidate = path.resolve(distRoot, relative)
  if (!candidate.startsWith(`${distRoot}${path.sep}`) && candidate !== distRoot) return null
  if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return candidate
  return path.join(distRoot, "index.html")
}

const serveRequest = async (request, response) => {
  try {
    if (request.url.startsWith("/api/")) {
      await proxyAPI(request, response)
      return
    }
    const asset = resolveAsset(request.url)
    if (!asset) {
      response.writeHead(403)
      response.end("Forbidden")
      return
    }
    response.setHeader("Content-Type", contentTypes[path.extname(asset)] || "application/octet-stream")
    response.setHeader("Cache-Control", asset.endsWith("index.html") ? "no-store" : "public, max-age=31536000, immutable")
    fs.createReadStream(asset).pipe(response)
  } catch (error) {
    response.writeHead(502, { "Content-Type": "application/json" })
    response.end(JSON.stringify({ error: error.message }))
  }
}

const startWebServer = () => new Promise((resolve, reject) => {
  webServer = http.createServer((request, response) => void serveRequest(request, response))
  webServer.once("error", reject)
  webServer.listen(0, "127.0.0.1", () => resolve(webServer.address().port))
})

const createWindow = async port => {
  mainWindow = new BrowserWindow({
    width: 1500,
    height: 960,
    minWidth: 980,
    minHeight: 680,
    show: false,
    title: "Reelmark",
    backgroundColor: "#f6f3ed",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  })
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://") || url.startsWith("http://")) void shell.openExternal(url)
    return { action: "deny" }
  })
  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith(`http://127.0.0.1:${port}`)) {
      event.preventDefault()
      if (url.startsWith("https://") || url.startsWith("http://")) void shell.openExternal(url)
    }
  })
  mainWindow.once("ready-to-show", () => mainWindow.show())
  await mainWindow.loadURL(`http://127.0.0.1:${port}`)
  if (!mainWindow.isVisible()) mainWindow.show()
}

const launch = async () => {
  try {
    await startBackend()
    const port = await startWebServer()
    await createWindow(port)
  } catch (error) {
    const logs = app.getPath("logs")
    fs.mkdirSync(logs, { recursive: true })
    fs.appendFileSync(path.join(logs, "reelmark.log"), `${new Date().toISOString()} ${error.stack || error.message}\n`)
    dialog.showErrorBox("Reelmark could not open", error.message)
    app.quit()
  }
}

app.whenReady().then(launch)

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0 && webServer) {
    const port = webServer.address().port
    void createWindow(port)
  }
})

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit()
})

app.on("before-quit", () => {
  if (backend && backend.exitCode === null) backend.kill("SIGTERM")
  if (webServer) webServer.close()
})
