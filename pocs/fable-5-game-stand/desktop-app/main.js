const { app, BrowserWindow, Menu, shell } = require("electron")
const { spawn } = require("child_process")
const fs = require("fs")
const http = require("http")
const net = require("net")
const path = require("path")

const sourceFile = path.join(process.resourcesPath, "source-path")
const projectDirectory = fs.existsSync(sourceFile)
  ? fs.readFileSync(sourceFile, "utf8").trim()
  : path.resolve(__dirname, "..")
const desktopDirectory = path.join(projectDirectory, "desktop-app")
const python = path.join(desktopDirectory, ".venv", "bin", "python")
const pidFile = path.join(desktopDirectory, ".desktop.pid")
let backend
let backendOutput = ""
let mainWindow

function findPort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.once("error", reject)
    server.listen(0, "127.0.0.1", () => {
      const port = server.address().port
      server.close(() => resolve(port))
    })
  })
}

function waitForBackend(port, attempts = 60) {
  return new Promise((resolve, reject) => {
    const check = remaining => {
      const request = http.get(`http://127.0.0.1:${port}/api/games`, response => {
        response.resume()
        if (response.statusCode === 200) {
          resolve()
          return
        }
        retry(remaining)
      })
      request.on("error", () => retry(remaining))
      request.setTimeout(1000, () => request.destroy())
    }
    const retry = remaining => {
      if (remaining <= 1) {
        reject(new Error(backendOutput.trim() || "The local service did not start"))
        return
      }
      setTimeout(() => check(remaining - 1), 500)
    }
    check(attempts)
  })
}

async function startBackend() {
  const port = await findPort()
  backend = spawn(python, [path.join(projectDirectory, "app.py")], {
    cwd: projectDirectory,
    env: { ...process.env, GAME_STAND_PORT: String(port) },
    stdio: ["ignore", "ignore", "pipe"]
  })
  backend.stderr.on("data", data => {
    backendOutput = `${backendOutput}${data}`.slice(-4000)
  })
  await waitForBackend(port)
  return port
}

function createWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 960,
    minHeight: 700,
    backgroundColor: "#f7f1e3",
    show: false,
    title: "Game Stand",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  })
  mainWindow.loadURL(`http://127.0.0.1:${port}`)
  mainWindow.once("ready-to-show", () => mainWindow.show())
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("http://") || url.startsWith("https://")) shell.openExternal(url)
    return { action: "deny" }
  })
}

function stopBackend() {
  if (backend && !backend.killed) backend.kill("SIGTERM")
  backend = undefined
}

function removePidFile() {
  if (fs.existsSync(pidFile)) fs.unlinkSync(pidFile)
}

app.setName("Game Stand")
Menu.setApplicationMenu(null)
fs.writeFileSync(pidFile, String(process.pid))

app.whenReady().then(async () => {
  try {
    const port = await startBackend()
    createWindow(port)
  } catch (error) {
    const { dialog } = require("electron")
    dialog.showErrorBox("Game Stand could not start", error.message)
    app.quit()
  }
})

app.on("window-all-closed", () => {
  app.quit()
})

app.on("before-quit", stopBackend)
app.on("will-quit", removePidFile)
process.on("exit", removePidFile)
