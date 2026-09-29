const { app, BrowserWindow, Menu, ipcMain, screen, shell, clipboard, net } = require("electron");
const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const APP_NAME = "SV Map";
const USER_AGENT = "sv-map/1.0 (+https://github.com/diegopacheco/ai-playground)";

function readResource(name) {
  const file = path.join(process.resourcesPath || "", name);
  return fs.existsSync(file) ? fs.readFileSync(file, "utf8").trim() : null;
}

const ROOT = readResource("source-path") || path.resolve(__dirname, "..");
const NODE_DIR = readResource("node-dir");
const PORT = fs.readFileSync(path.join(ROOT, "scripts/ports.env"), "utf8").match(/^WEB=(\d+)/m)[1];
const APP_URL = `http://localhost:${PORT}`;
const STATE_FILE = () => path.join(app.getPath("userData"), "window-state.json");
const ENV = { ...process.env, PATH: [NODE_DIR, "/opt/homebrew/bin", "/usr/local/bin", process.env.PATH].filter(Boolean).join(":") };

let win = null;
let restoreBounds = null;
let stopped = false;

app.setName(APP_NAME);
app.userAgentFallback = USER_AGENT;

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!win) return;
    if (win.isMinimized()) win.restore();
    win.show();
    win.focus();
  });
  app.whenReady().then(boot);
}

function loadState() {
  try {
    const saved = JSON.parse(fs.readFileSync(STATE_FILE(), "utf8"));
    const visible = screen.getAllDisplays().some(d => {
      const a = d.workArea;
      return saved.x >= a.x - 50 && saved.y >= a.y - 50 && saved.x < a.x + a.width && saved.y < a.y + a.height;
    });
    return visible ? saved : {};
  } catch {
    return {};
  }
}

function saveState() {
  if (!win || win.isDestroyed()) return;
  const bounds = restoreBounds || win.getNormalBounds();
  const state = { ...bounds, fullscreen: win.isFullScreen(), maximized: Boolean(restoreBounds) };
  fs.mkdirSync(path.dirname(STATE_FILE()), { recursive: true });
  fs.writeFileSync(STATE_FILE(), JSON.stringify(state));
}

function toggleMaximize() {
  if (win.isFullScreen()) return;
  if (restoreBounds) {
    win.setBounds(restoreBounds, true);
    restoreBounds = null;
  } else {
    restoreBounds = win.getBounds();
    win.setBounds(screen.getDisplayMatching(restoreBounds).workArea, true);
  }
  saveState();
}

function createWindow() {
  const saved = loadState();
  win = new BrowserWindow({
    x: saved.x,
    y: saved.y,
    width: saved.width || 1400,
    height: saved.height || 900,
    minWidth: 760,
    minHeight: 520,
    title: APP_NAME,
    titleBarStyle: "hiddenInset",
    backgroundColor: "#f8fafc",
    show: false,
    webPreferences: { preload: path.join(__dirname, "preload.cjs"), contextIsolation: true, nodeIntegration: false }
  });
  if (saved.maximized) {
    restoreBounds = { x: saved.x, y: saved.y, width: saved.width, height: saved.height };
    win.setBounds(screen.getDisplayMatching(restoreBounds).workArea);
  }
  if (saved.fullscreen) win.setFullScreen(true);
  win.once("ready-to-show", () => win.show());
  let timer = null;
  const queueSave = () => { clearTimeout(timer); timer = setTimeout(saveState, 300); };
  win.on("move", queueSave);
  win.on("resize", queueSave);
  win.on("enter-full-screen", saveState);
  win.on("leave-full-screen", saveState);
  win.on("close", saveState);
  win.on("closed", () => { win = null; });
  win.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: "deny" }; });
  win.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith(APP_URL) && !url.startsWith("file:")) { event.preventDefault(); shell.openExternal(url); }
  });
}

function stamp() {
  const d = new Date();
  const p = n => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

async function captureWindow() {
  if (!win) return;
  const image = await win.webContents.capturePage();
  const file = path.join(os.homedir(), "Desktop", `sv-map-${stamp()}.png`);
  fs.writeFileSync(file, image.toPNG());
  clipboard.writeImage(image);
  win.webContents.send("toast", `Screenshot saved to Desktop as ${path.basename(file)} and copied`);
}

function buildMenu() {
  const zoom = delta => () => { if (win) win.webContents.setZoomLevel(win.webContents.getZoomLevel() + delta); };
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: APP_NAME, submenu: [{ role: "about" }, { type: "separator" }, { role: "hide" }, { role: "hideOthers" }, { type: "separator" }, { role: "quit" }] },
    { label: "Edit", submenu: [{ role: "undo" }, { role: "redo" }, { type: "separator" }, { role: "cut" }, { role: "copy" }, { role: "paste" }, { role: "selectAll" }] },
    { label: "View", submenu: [
      { label: "Zoom In", accelerator: "CmdOrCtrl+=", click: zoom(0.5) },
      { label: "Zoom In", accelerator: "CmdOrCtrl+Plus", click: zoom(0.5), visible: false },
      { label: "Zoom Out", accelerator: "CmdOrCtrl+-", click: zoom(-0.5) },
      { label: "Actual Size", accelerator: "CmdOrCtrl+0", click: () => win && win.webContents.setZoomLevel(0) },
      { type: "separator" },
      { label: "Toggle Full Screen", accelerator: "CmdOrCtrl+Shift+Enter", click: () => win && win.setFullScreen(!win.isFullScreen()) },
      { label: "Screenshot", accelerator: "CmdOrCtrl+P", click: captureWindow },
      { type: "separator" },
      { label: "Reload", accelerator: "CmdOrCtrl+Shift+R", click: () => win && win.reload() },
      { role: "toggleDevTools" }
    ] },
    { label: "Window", submenu: [{ role: "minimize" }, { role: "close" }] }
  ]));
}

function sendBoot(id, status, detail) {
  if (win && !win.isDestroyed()) win.webContents.send("boot", { id, status, detail });
}

function runScript(name) {
  return new Promise(resolve => {
    const child = spawn("bash", [path.join(ROOT, "scripts", name)], { cwd: ROOT, env: ENV });
    let output = "";
    child.stdout.on("data", chunk => { output += chunk; });
    child.stderr.on("data", chunk => { output += chunk; });
    child.on("close", code => resolve({ code, output }));
  });
}

async function fetchOk(url) {
  try {
    const res = await net.fetch(url, { headers: { "User-Agent": USER_AGENT } });
    return res.ok ? res : null;
  } catch {
    return null;
  }
}

async function bootServices() {
  const node = spawnSync("node", ["--version"], { env: ENV, encoding: "utf8" });
  if (node.status !== 0) { sendBoot("node", "fail", "node not found on PATH"); return false; }
  sendBoot("node", "ok", node.stdout.trim());

  sendBoot("server", "wait", "running scripts/start-all.sh");
  const started = await runScript("start-all.sh");
  const health = await fetchOk(`${APP_URL}/api/health`);
  if (started.code !== 0 || !health) { sendBoot("server", "fail", started.output.trim().split("\n").pop() || "server did not start"); return false; }
  sendBoot("server", "ok", APP_URL);

  const { companies } = await health.json();
  sendBoot("data", "ok", `${companies} companies loaded`);

  const logos = fs.existsSync(path.join(ROOT, "data/logos")) ? fs.readdirSync(path.join(ROOT, "data/logos")).length : 0;
  sendBoot("logos", logos ? "ok" : "warn", `${logos} logos on disk`);

  sendBoot("tiles", "wait", "tile.openstreetmap.org");
  const tile = await fetchOk("https://tile.openstreetmap.org/10/163/396.png");
  sendBoot("tiles", tile ? "ok" : "warn", tile ? "reachable" : "offline, map tiles will be blank");
  return true;
}

async function boot() {
  buildMenu();
  ipcMain.on("toggle-maximize", () => win && toggleMaximize());
  createWindow();
  await win.loadFile(path.join(__dirname, "boot.html"));
  if (await bootServices()) {
    setTimeout(() => win && win.loadURL(APP_URL), 700);
  }
}

function stopServices() {
  if (stopped) return;
  stopped = true;
  spawnSync("bash", [path.join(ROOT, "scripts/stop-all.sh")], { cwd: ROOT, env: ENV, timeout: 20000 });
}

app.on("before-quit", stopServices);
app.on("window-all-closed", () => app.quit());
