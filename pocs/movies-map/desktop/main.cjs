const { app, BrowserWindow, Menu, ipcMain, screen, shell, clipboard, session } = require("electron")
const { spawn, spawnSync } = require("node:child_process")
const fs = require("node:fs")
const os = require("node:os")
const path = require("node:path")

const CONFIG_FILE = path.join(__dirname, "config.json")
const config = fs.existsSync(CONFIG_FILE) ? JSON.parse(fs.readFileSync(CONFIG_FILE, "utf8")) : {}
const ROOT = config.root || path.resolve(__dirname, "..")
const ENV = { ...process.env, PATH: [config.nodeDir, "/opt/homebrew/bin", "/usr/local/bin", process.env.PATH].filter(Boolean).join(":") }
const STATE_FILE = path.join(app.getPath("userData"), "window-state.json")
const TILE_URL = "https://tile.openstreetmap.org/13/1310/3166.png"

let win = null
let restoreBounds = null
let stopping = false

function port() {
  const text = fs.readFileSync(path.join(ROOT, "scripts", "ports.env"), "utf8")
  const match = /^api=(\d+)/m.exec(text)
  return Number(match ? match[1] : 4545)
}

function readState() {
  try {
    return JSON.parse(fs.readFileSync(STATE_FILE, "utf8"))
  } catch {
    return {}
  }
}

function visible(bounds) {
  return bounds && screen.getAllDisplays().some(d => {
    const a = d.workArea
    return bounds.x < a.x + a.width - 80 && bounds.x + bounds.width > a.x + 80 && bounds.y >= a.y - 20 && bounds.y < a.y + a.height - 80
  })
}

function saveState() {
  if (!win || win.isDestroyed()) return
  const bounds = restoreBounds || win.getNormalBounds()
  fs.mkdirSync(path.dirname(STATE_FILE), { recursive: true })
  fs.writeFileSync(STATE_FILE, JSON.stringify({ bounds, maximized: Boolean(restoreBounds), fullScreen: win.isFullScreen() }))
}

function toggleMaximize() {
  if (!win || win.isFullScreen()) return
  if (restoreBounds) {
    win.setBounds(restoreBounds, true)
    restoreBounds = null
  } else {
    restoreBounds = win.getBounds()
    win.setBounds(screen.getDisplayMatching(restoreBounds).workArea, true)
  }
  saveState()
}

function send(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload)
}

function run(script) {
  return new Promise(resolve => {
    const child = spawn("bash", [path.join(ROOT, "scripts", script)], { cwd: ROOT, env: ENV })
    let output = ""
    child.stdout.on("data", d => (output += d))
    child.stderr.on("data", d => (output += d))
    child.on("close", code => resolve({ code, output }))
  })
}

function log(line) {
  try {
    fs.mkdirSync(path.join(ROOT, ".run", "logs"), { recursive: true })
    fs.appendFileSync(path.join(ROOT, ".run", "logs", "desktop.log"), `${new Date().toISOString()} ${line}\n`)
  } catch {}
}

async function startApi() {
  let last = ""
  for (let attempt = 1; attempt <= 3; attempt++) {
    const started = await run("start-all.sh")
    last = started.output.trim().split("\n").pop() || `start-all.sh exited ${started.code}`
    log(`start-all attempt ${attempt} code ${started.code}: ${started.output.trim().replace(/\n/g, " | ")}`)
    const health = started.code === 0 && await fetch(`http://localhost:${port()}/api/health`).then(r => r.ok).catch(() => false)
    if (health) return null
    if (started.code === 0) last = "API did not answer /api/health"
    await new Promise(r => setTimeout(r, 1000))
  }
  return last
}

async function bootServices() {
  const step = (id, status, detail) => {
    log(`boot ${id} ${status} ${detail || ""}`)
    send("boot-step", { id, status, detail })
  }
  step("node", "loading")
  const node = spawnSync("node", ["-v"], { env: ENV, encoding: "utf8" })
  if (node.status !== 0) return step("node", "failed", "Node.js was not found in PATH")
  step("node", "ready", `Node.js ${node.stdout.trim()}`)

  step("data", "loading")
  const dataFile = path.join(ROOT, "data", "movies.json")
  if (!fs.existsSync(dataFile)) return step("data", "failed", "Run scripts/setup.sh to build data/movies.json")
  const { movies } = JSON.parse(fs.readFileSync(dataFile, "utf8"))
  step("data", "ready", `${movies.length} titles, ${movies.reduce((n, m) => n + m.locations.length, 0)} SF locations`)

  step("api", "loading")
  const error = await startApi()
  if (error) return step("api", "failed", error)
  step("api", "ready", `API ready on port ${port()}`)

  step("tiles", "loading")
  const tiles = await fetch(TILE_URL, { method: "HEAD", headers: { "User-Agent": "MoviesMap/1.0" } }).then(r => r.ok).catch(() => false)
  step("tiles", tiles ? "ready" : "failed", tiles ? "Map tiles reachable" : "Map tiles offline, posters still load")

  await session.defaultSession.clearCache()
  setTimeout(() => win && !win.isDestroyed() && win.loadURL(`http://localhost:${port()}/`), 500)
}

async function capture() {
  const image = await win.webContents.capturePage()
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)
  const file = path.join(os.homedir(), "Desktop", `Movies Map ${stamp}.png`)
  fs.writeFileSync(file, image.toPNG())
  clipboard.writeImage(image)
  send("toast", "Screenshot saved to Desktop and copied to clipboard")
}

function buildMenu() {
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: "Movies Map", submenu: [{ role: "about" }, { type: "separator" }, { role: "hide" }, { role: "hideOthers" }, { type: "separator" }, { role: "quit" }] },
    { label: "Edit", submenu: [{ role: "undo" }, { role: "redo" }, { type: "separator" }, { role: "cut" }, { role: "copy" }, { role: "paste" }, { role: "selectAll" }] },
    {
      label: "View",
      submenu: [
        { role: "zoomIn", accelerator: "CmdOrCtrl+=" },
        { role: "zoomIn", accelerator: "CmdOrCtrl+Plus", visible: false },
        { role: "zoomOut", accelerator: "CmdOrCtrl+-" },
        { role: "resetZoom", accelerator: "CmdOrCtrl+0" },
        { type: "separator" },
        { label: "Toggle Full Screen", accelerator: "CmdOrCtrl+Shift+Enter", click: () => win && win.setFullScreen(!win.isFullScreen()) },
        { label: "Screenshot", accelerator: "CmdOrCtrl+P", click: () => win && capture() },
        { type: "separator" },
        { role: "reload" },
        { role: "toggleDevTools" }
      ]
    },
    { label: "Window", submenu: [{ role: "minimize" }, { label: "Maximize or Restore", click: toggleMaximize }, { role: "close" }] }
  ]))
}

function createWindow() {
  const saved = readState()
  const bounds = visible(saved.bounds) ? saved.bounds : { width: 1440, height: 920 }
  win = new BrowserWindow({
    ...bounds,
    minWidth: 900,
    minHeight: 600,
    title: "Movies Map",
    titleBarStyle: "hiddenInset",
    backgroundColor: "#faf7f2",
    show: false,
    webPreferences: { preload: path.join(__dirname, "preload.cjs"), contextIsolation: true, nodeIntegration: false }
  })
  if (saved.maximized) {
    restoreBounds = bounds
    win.setBounds(screen.getDisplayMatching(bounds).workArea)
  }
  win.once("ready-to-show", () => {
    win.show()
    if (saved.fullScreen) win.setFullScreen(true)
  })
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: "deny" }
  })
  win.webContents.on("will-navigate", (e, url) => {
    if (!url.startsWith(`http://localhost:${port()}`) && !url.startsWith("file:")) {
      e.preventDefault()
      shell.openExternal(url)
    }
  })
  win.on("moved", saveState)
  win.on("resize", saveState)
  win.on("enter-full-screen", saveState)
  win.on("leave-full-screen", saveState)
  win.on("close", saveState)
  win.loadFile(path.join(__dirname, "boot.html"))
  win.webContents.on("did-finish-load", () => {
    if (win.webContents.getURL().endsWith("boot.html")) bootServices()
  })
}

if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.setName("Movies Map")
  app.on("second-instance", () => {
    if (!win) return
    if (win.isMinimized()) win.restore()
    win.show()
    win.focus()
  })
  ipcMain.on("toggle-maximize", toggleMaximize)
  ipcMain.on("retry-boot", () => win && win.loadFile(path.join(__dirname, "boot.html")))
  app.whenReady().then(() => {
    buildMenu()
    createWindow()
  })
  app.on("activate", () => {
    if (!win || win.isDestroyed()) createWindow()
  })
  app.on("window-all-closed", () => app.quit())
  app.on("before-quit", e => {
    if (stopping) return
    stopping = true
    e.preventDefault()
    saveState()
    run("stop-all.sh").then(() => app.quit())
  })
}
