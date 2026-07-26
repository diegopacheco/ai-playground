const desktop = document.querySelector("#desktop")
const desktopIcons = document.querySelector("#desktop-icons")
const windowLayer = document.querySelector("#window-layer")
const taskButtons = document.querySelector("#task-buttons")
const startMenu = document.querySelector("#start-menu")
const startButton = document.querySelector("#start-button")
const startApps = document.querySelector("#start-apps")
const contextMenu = document.querySelector("#context-menu")
const shutdownScreen = document.querySelector("#shutdown-screen")

const appCatalog = {
  files: { name: "My Computer", icon: "🖥️", description: "Browse files and folders", size: [760, 500] },
  notepad: { name: "Notepad", icon: "📝", description: "Write a text note", size: [620, 430] },
  paint: { name: "Luma Paint", icon: "🎨", description: "Draw and save pictures", size: [780, 560] },
  terminal: { name: "Safe Terminal", icon: "⬛", description: "Use protected commands", size: [680, 430] },
  browser: { name: "Luma Explorer", icon: "🌐", description: "Browse the web", size: [820, 560] },
  images: { name: "Picture Viewer", icon: "🖼️", description: "View five local pictures", size: [760, 520] },
  videos: { name: "Video Player", icon: "🎞️", description: "Play two local animations", size: [760, 520] },
  clock: { name: "Clock", icon: "🕰️", description: "View local time", size: [390, 430] },
  settings: { name: "Desktop Properties", icon: "⚙️", description: "Personalize LumaOS", size: [700, 470] },
  help: { name: "Help & Support", icon: "❔", description: "Learn the desktop", size: [610, 450] },
  recycle: { name: "Recycle Bin", icon: "🗑️", description: "Stored removed items", size: [540, 380] }
}

const desktopApps = ["files", "notepad", "paint", "terminal", "browser", "images", "videos", "clock", "settings", "recycle"]
const startAppKeys = ["browser", "notepad", "paint", "terminal", "files", "clock"]
const wallpaperNames = ["Emerald Hills", "Moonlit Tide", "Amber Dunes", "Alpine Dawn", "Glass Current"]
const blockedWords = ["rm", "rmdir", "del", "erase", "format", "mkfs", "shutdown", "reboot", "kill", "sudo", "su", "chmod", "chown", "dd", "curl", "wget", "eval", "exec", "powershell"]
const forbiddenOperators = /[;&|><`$\\]/
const appState = {
  topZ: 20,
  windowCount: 0,
  currentWallpaper: 1,
  customWallpaper: "",
  terminalHistory: [],
  terminalHistoryIndex: 0,
  videoLoops: new Map()
}

const defaultFs = {
  type: "folder",
  children: {
    Documents: {
      type: "folder",
      children: {
        "Welcome.txt": {
          type: "file",
          content: "Welcome to LumaOS 2003!\n\nOpen apps from the desktop or Start menu.\nYour notes and folders stay in this browser."
        },
        "Things to try.txt": {
          type: "file",
          content: "Draw in Luma Paint\nVisit the picture collection\nChange the wallpaper\nTry help in Safe Terminal"
        }
      }
    },
    Pictures: { type: "folder", children: {} },
    Videos: { type: "folder", children: {} },
    "Shared Files": { type: "folder", children: {} }
  }
}

let virtualFs = loadJson("lumaos-fs", defaultFs)

function clone(value) {
  return JSON.parse(JSON.stringify(value))
}

function loadJson(key, fallback) {
  try {
    const saved = localStorage.getItem(key)
    return saved ? JSON.parse(saved) : clone(fallback)
  } catch {
    return clone(fallback)
  }
}

function saveFs() {
  try {
    localStorage.setItem("lumaos-fs", JSON.stringify(virtualFs))
  } catch {
    toast("Storage unavailable", "Changes will last until this page closes.")
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;")
}

function safeImageUrl(value) {
  const trimmed = String(value).trim()
  if (/^https?:\/\/\S+$/i.test(trimmed) || /^data:image\/(png|jpeg|webp|gif);base64,/i.test(trimmed)) {
    return trimmed.replaceAll('"', "%22").replaceAll("'", "%27")
  }
  return ""
}

function renderDesktopIcons() {
  desktopIcons.innerHTML = desktopApps.map(key => {
    const app = appCatalog[key]
    return `<button class="desktop-icon" data-app="${key}" title="Open ${escapeHtml(app.name)}"><span class="icon-art">${app.icon}</span><span>${escapeHtml(app.name)}</span></button>`
  }).join("")
}

function renderStartApps() {
  startApps.innerHTML = startAppKeys.map(key => {
    const app = appCatalog[key]
    return `<button class="start-app" data-open="${key}"><span>${app.icon}</span><div><strong>${escapeHtml(app.name)}</strong><small>${escapeHtml(app.description)}</small></div></button>`
  }).join("")
}

function createWindow(key, options = {}) {
  const app = appCatalog[key]
  const id = `window-${++appState.windowCount}`
  const width = Math.min(options.width || app.size[0], window.innerWidth - 24)
  const height = Math.min(options.height || app.size[1], window.innerHeight - 58)
  const offset = (appState.windowCount * 24) % 190
  const left = Math.max(5, Math.min(115 + offset, window.innerWidth - width - 8))
  const top = Math.max(5, Math.min(38 + offset / 2, window.innerHeight - height - 48))
  const windowEl = document.createElement("section")
  windowEl.className = "os-window"
  windowEl.id = id
  windowEl.dataset.app = key
  windowEl.style.width = `${width}px`
  windowEl.style.height = `${height}px`
  windowEl.style.left = `${left}px`
  windowEl.style.top = `${top}px`
  windowEl.innerHTML = `
    <header class="titlebar">
      <span class="title-icon">${options.icon || app.icon}</span>
      <span class="title-text">${escapeHtml(options.title || app.name)}</span>
      <div class="window-controls">
        <button class="minimize-window" aria-label="Minimize">_</button>
        <button class="maximize-window" aria-label="Maximize">□</button>
        <button class="close-window" aria-label="Close">×</button>
      </div>
    </header>
    ${options.menu === false ? "" : `<nav class="menu-strip"><span>File</span><span>Edit</span><span>View</span><span>Help</span></nav>`}
    <div class="window-content"></div>
    ${options.status === false ? "" : `<footer class="statusbar"><span>${escapeHtml(options.status || "Ready")}</span><span>LumaOS</span></footer>`}
  `
  windowLayer.append(windowEl)
  const taskButton = document.createElement("button")
  taskButton.className = "task-button"
  taskButton.dataset.windowId = id
  taskButton.innerHTML = `<span>${options.icon || app.icon}</span><span>${escapeHtml(options.title || app.name)}</span>`
  taskButtons.append(taskButton)
  bindWindow(windowEl, taskButton)
  focusWindow(windowEl)
  return windowEl
}

function bindWindow(windowEl, taskButton) {
  const titlebar = windowEl.querySelector(".titlebar")
  const minimize = windowEl.querySelector(".minimize-window")
  const maximize = windowEl.querySelector(".maximize-window")
  const close = windowEl.querySelector(".close-window")

  windowEl.addEventListener("pointerdown", () => focusWindow(windowEl))
  titlebar.addEventListener("dblclick", () => toggleMaximize(windowEl))

  titlebar.addEventListener("pointerdown", event => {
    if (event.target.closest("button") || windowEl.classList.contains("maximized")) return
    event.preventDefault()
    focusWindow(windowEl)
    const startX = event.clientX
    const startY = event.clientY
    const startLeft = windowEl.offsetLeft
    const startTop = windowEl.offsetTop
    titlebar.setPointerCapture(event.pointerId)

    const move = moveEvent => {
      const maxLeft = Math.max(0, windowLayer.clientWidth - 90)
      const maxTop = Math.max(0, windowLayer.clientHeight - 35)
      windowEl.style.left = `${Math.max(0, Math.min(maxLeft, startLeft + moveEvent.clientX - startX))}px`
      windowEl.style.top = `${Math.max(0, Math.min(maxTop, startTop + moveEvent.clientY - startY))}px`
    }

    const end = () => {
      titlebar.removeEventListener("pointermove", move)
      titlebar.removeEventListener("pointerup", end)
      titlebar.removeEventListener("pointercancel", end)
    }

    titlebar.addEventListener("pointermove", move)
    titlebar.addEventListener("pointerup", end)
    titlebar.addEventListener("pointercancel", end)
  })

  minimize.addEventListener("click", () => {
    windowEl.classList.add("minimized")
    taskButton.classList.remove("active")
  })

  maximize.addEventListener("click", () => toggleMaximize(windowEl))
  close.addEventListener("click", () => closeWindow(windowEl))

  taskButton.addEventListener("click", () => {
    if (windowEl.classList.contains("minimized")) {
      windowEl.classList.remove("minimized")
      focusWindow(windowEl)
    } else if (taskButton.classList.contains("active")) {
      windowEl.classList.add("minimized")
      taskButton.classList.remove("active")
    } else {
      focusWindow(windowEl)
    }
  })
}

function focusWindow(windowEl) {
  document.querySelectorAll(".os-window").forEach(item => item.classList.add("inactive"))
  document.querySelectorAll(".task-button").forEach(item => item.classList.remove("active"))
  windowEl.classList.remove("inactive")
  windowEl.classList.remove("minimized")
  windowEl.style.zIndex = String(++appState.topZ)
  document.querySelector(`[data-window-id="${windowEl.id}"]`)?.classList.add("active")
}

function toggleMaximize(windowEl) {
  windowEl.classList.toggle("maximized")
  focusWindow(windowEl)
}

function closeWindow(windowEl) {
  const loop = appState.videoLoops.get(windowEl.id)
  if (loop) cancelAnimationFrame(loop)
  appState.videoLoops.delete(windowEl.id)
  document.querySelector(`[data-window-id="${windowEl.id}"]`)?.remove()
  windowEl.remove()
  const remaining = [...document.querySelectorAll(".os-window:not(.minimized)")]
  if (remaining.length) focusWindow(remaining.at(-1))
}

function updateWindowTitle(windowEl, title) {
  windowEl.querySelector(".title-text").textContent = title
  const task = document.querySelector(`[data-window-id="${windowEl.id}"] span:last-child`)
  if (task) task.textContent = title
}

function openApp(key, options = {}) {
  closeStartMenu()
  contextMenu.hidden = true
  const handlers = {
    files: openFiles,
    notepad: openNotepad,
    paint: openPaint,
    terminal: openTerminal,
    browser: openBrowser,
    images: openImages,
    videos: openVideos,
    clock: openClock,
    settings: openSettings,
    help: openHelp,
    recycle: openRecycle
  }
  handlers[key]?.(options)
}

function openFiles(options = {}) {
  const windowEl = createWindow("files", { status: "Local browser storage" })
  const content = windowEl.querySelector(".window-content")
  let path = Array.isArray(options.path) ? options.path : []

  content.innerHTML = `
    <div class="explorer">
      <aside class="explorer-sidebar">
        <div class="side-card">
          <h3>File and Folder Tasks</h3>
          <button data-file-action="folder">Make a new folder</button>
          <button data-file-action="note">Make a new text note</button>
        </div>
        <div class="side-card">
          <h3>Other Places</h3>
          <button data-path="">Desktop root</button>
          <button data-path="Documents">My Documents</button>
          <button data-path="Pictures">My Pictures</button>
          <button data-path="Videos">My Videos</button>
        </div>
        <div class="side-card">
          <h3>Details</h3>
          <button type="button">Items are saved locally</button>
        </div>
      </aside>
      <div class="explorer-main">
        <div class="explorer-address">
          <button class="xp-button" data-file-action="up">⬅ Back</button>
          <strong>Address</strong>
          <input readonly aria-label="Current folder">
          <button class="xp-button" data-file-action="folder">New folder</button>
        </div>
        <div class="file-grid"></div>
      </div>
    </div>
  `

  const grid = content.querySelector(".file-grid")
  const address = content.querySelector(".explorer-address input")

  const getNode = () => {
    let node = virtualFs
    for (const part of path) {
      node = node.children?.[part]
      if (!node || node.type !== "folder") {
        path = []
        return virtualFs
      }
    }
    return node
  }

  const render = () => {
    const node = getNode()
    const entries = Object.entries(node.children || {}).sort((a, b) => {
      if (a[1].type !== b[1].type) return a[1].type === "folder" ? -1 : 1
      return a[0].localeCompare(b[0])
    })
    address.value = `LumaOS:\\${path.join("\\") || "My Computer"}`
    updateWindowTitle(windowEl, path.length ? `${path.at(-1)} - My Computer` : "My Computer")
    grid.innerHTML = entries.length
      ? entries.map(([name, entry]) => `<button class="file-item" data-entry="${encodeURIComponent(name)}"><i>${entry.type === "folder" ? "📁" : "📄"}</i><span>${escapeHtml(name)}</span></button>`).join("")
      : `<div class="help-card"><strong>This folder is empty</strong>Create a folder or text note to add an item.</div>`
  }

  const createFolder = () => {
    const requested = prompt("Folder name")
    if (!requested) return
    const name = cleanFileName(requested)
    if (!name) {
      toast("Folder not created", "Use letters, numbers, spaces, dashes or underscores.")
      return
    }
    const node = getNode()
    if (node.children[name]) {
      toast("Name already in use", "Choose another folder name.")
      return
    }
    node.children[name] = { type: "folder", children: {} }
    saveFs()
    render()
  }

  const createNote = () => {
    const requested = prompt("Text note name", "New Note.txt")
    if (!requested) return
    let name = cleanFileName(requested)
    if (!name) return
    if (!name.toLowerCase().endsWith(".txt")) name += ".txt"
    const node = getNode()
    if (node.children[name]) {
      toast("Name already in use", "Choose another note name.")
      return
    }
    node.children[name] = { type: "file", content: "" }
    saveFs()
    render()
    openNotepad({ filePath: [...path, name] })
  }

  content.addEventListener("dblclick", event => {
    const item = event.target.closest("[data-entry]")
    if (!item) return
    const name = decodeURIComponent(item.dataset.entry)
    const entry = getNode().children[name]
    if (entry.type === "folder") {
      path.push(name)
      render()
    } else {
      openNotepad({ filePath: [...path, name] })
    }
  })

  content.addEventListener("click", event => {
    const action = event.target.closest("[data-file-action]")?.dataset.fileAction
    const goPath = event.target.closest("[data-path]")
    if (goPath) {
      path = goPath.dataset.path ? [goPath.dataset.path] : []
      render()
    }
    if (action === "up" && path.length) {
      path.pop()
      render()
    }
    if (action === "folder") createFolder()
    if (action === "note") createNote()
  })

  render()
}

function cleanFileName(value) {
  return String(value).trim().replace(/[^\w .-]/g, "").slice(0, 64)
}

function nodeAt(path) {
  let node = virtualFs
  for (const part of path) node = node.children?.[part]
  return node
}

function openNotepad(options = {}) {
  const filePath = options.filePath
  const fileNode = filePath ? nodeAt(filePath) : null
  const fileName = filePath?.at(-1) || "Untitled"
  const windowEl = createWindow("notepad", { title: `${fileName} - Notepad`, status: "Plain text" })
  const content = windowEl.querySelector(".window-content")
  content.innerHTML = `
    <div class="explorer-address">
      <button class="xp-button" data-note="save">💾 Save</button>
      <button class="xp-button" data-note="download">⬇ Download</button>
      <button class="xp-button" data-note="clear">Clear</button>
      <span data-note-state>${fileNode ? "Saved file" : "New note"}</span>
    </div>
    <textarea class="notepad" spellcheck="true" aria-label="Note text">${escapeHtml(fileNode?.content || options.content || "")}</textarea>
  `
  const textarea = content.querySelector("textarea")
  const state = content.querySelector("[data-note-state]")
  let currentPath = filePath

  const save = () => {
    if (!currentPath) {
      const requested = prompt("Save note as", "New Note.txt")
      if (!requested) return
      let name = cleanFileName(requested)
      if (!name) return
      if (!name.toLowerCase().endsWith(".txt")) name += ".txt"
      currentPath = ["Documents", name]
    }
    const parent = nodeAt(currentPath.slice(0, -1))
    if (!parent || parent.type !== "folder") return
    parent.children[currentPath.at(-1)] = { type: "file", content: textarea.value }
    saveFs()
    updateWindowTitle(windowEl, `${currentPath.at(-1)} - Notepad`)
    state.textContent = `Saved ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
    toast("Note saved", `Documents/${currentPath.at(-1)}`)
  }

  const download = () => {
    const blob = new Blob([textarea.value], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = currentPath?.at(-1) || "LumaOS Note.txt"
    link.click()
    URL.revokeObjectURL(url)
  }

  textarea.addEventListener("input", () => {
    state.textContent = `${textarea.value.length} characters`
  })
  textarea.addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault()
      save()
    }
  })
  content.addEventListener("click", event => {
    const action = event.target.closest("[data-note]")?.dataset.note
    if (action === "save") save()
    if (action === "download") download()
    if (action === "clear" && confirm("Clear this note?")) textarea.value = ""
  })
  setTimeout(() => textarea.focus(), 0)
}

function openPaint() {
  const windowEl = createWindow("paint", { menu: false, status: "Canvas 900 × 560" })
  const content = windowEl.querySelector(".window-content")
  content.innerHTML = `
    <div class="paint-app">
      <aside class="paint-tools">
        <button class="active" data-tool="pencil" title="Pencil">✏️</button>
        <button data-tool="brush" title="Brush">🖌️</button>
        <button data-tool="eraser" title="Eraser">▱</button>
        <button data-paint="undo" title="Undo">↶</button>
        <button data-paint="clear" title="Clear">⌫</button>
        <button data-paint="save" title="Save PNG">💾</button>
      </aside>
      <div class="paint-stage"><canvas width="900" height="560"></canvas></div>
      <div class="paint-palette">
        ${["#111111", "#ffffff", "#e53935", "#ff9800", "#ffd740", "#45a73b", "#0797c7", "#1565c0", "#6a45b8", "#e5489b"].map(color => `<button style="--swatch:${color}" data-color="${color}" aria-label="${color}"></button>`).join("")}
        <label>Size <input type="range" min="1" max="34" value="4" data-paint-size></label>
        <span data-paint-info>Pencil · black</span>
      </div>
    </div>
  `
  const canvas = content.querySelector("canvas")
  const ctx = canvas.getContext("2d", { willReadFrequently: true })
  const info = content.querySelector("[data-paint-info]")
  const sizeInput = content.querySelector("[data-paint-size]")
  const snapshots = []
  let drawing = false
  let tool = "pencil"
  let color = "#111111"

  ctx.fillStyle = "#ffffff"
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.lineCap = "round"
  ctx.lineJoin = "round"

  const position = event => {
    const rect = canvas.getBoundingClientRect()
    return {
      x: (event.clientX - rect.left) * canvas.width / rect.width,
      y: (event.clientY - rect.top) * canvas.height / rect.height
    }
  }

  canvas.addEventListener("pointerdown", event => {
    drawing = true
    snapshots.push(ctx.getImageData(0, 0, canvas.width, canvas.height))
    if (snapshots.length > 15) snapshots.shift()
    const point = position(event)
    ctx.beginPath()
    ctx.moveTo(point.x, point.y)
    canvas.setPointerCapture(event.pointerId)
  })

  canvas.addEventListener("pointermove", event => {
    if (!drawing) return
    const point = position(event)
    ctx.globalCompositeOperation = tool === "eraser" ? "destination-out" : "source-over"
    ctx.strokeStyle = tool === "eraser" ? "#ffffff" : color
    ctx.lineWidth = Number(sizeInput.value) * (tool === "brush" ? 2.4 : tool === "eraser" ? 2.8 : 1)
    ctx.lineTo(point.x, point.y)
    ctx.stroke()
  })

  const stop = () => {
    drawing = false
    ctx.closePath()
  }
  canvas.addEventListener("pointerup", stop)
  canvas.addEventListener("pointercancel", stop)

  content.addEventListener("click", event => {
    const toolButton = event.target.closest("[data-tool]")
    const colorButton = event.target.closest("[data-color]")
    const action = event.target.closest("[data-paint]")?.dataset.paint
    if (toolButton) {
      tool = toolButton.dataset.tool
      content.querySelectorAll("[data-tool]").forEach(button => button.classList.toggle("active", button === toolButton))
      info.textContent = `${tool[0].toUpperCase()}${tool.slice(1)} · ${color}`
    }
    if (colorButton) {
      color = colorButton.dataset.color
      info.textContent = `${tool[0].toUpperCase()}${tool.slice(1)} · ${color}`
    }
    if (action === "undo" && snapshots.length) ctx.putImageData(snapshots.pop(), 0, 0)
    if (action === "clear" && confirm("Clear the canvas?")) {
      snapshots.push(ctx.getImageData(0, 0, canvas.width, canvas.height))
      ctx.globalCompositeOperation = "source-over"
      ctx.fillStyle = "#ffffff"
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    }
    if (action === "save") {
      const link = document.createElement("a")
      link.download = `Luma-Paint-${Date.now()}.png`
      link.href = canvas.toDataURL("image/png")
      link.click()
    }
  })
}

function openTerminal() {
  const windowEl = createWindow("terminal", { menu: false, status: "Protected command mode" })
  const content = windowEl.querySelector(".window-content")
  content.innerHTML = `
    <div class="terminal-app">
      <div class="terminal-output">LumaOS Safe Terminal 1.0
Type "help" to see allowed commands.
Protected mode blocks destructive commands and shell operators.

</div>
      <div class="terminal-line">
        <span class="terminal-prompt">diego@luma:~$&nbsp;</span>
        <input class="terminal-input" autocomplete="off" autocapitalize="off" spellcheck="false" aria-label="Terminal command">
      </div>
    </div>
  `
  const terminal = content.querySelector(".terminal-app")
  const output = content.querySelector(".terminal-output")
  const input = content.querySelector(".terminal-input")
  let cwd = []

  const write = value => {
    output.textContent += `${value}\n`
    terminal.scrollTop = terminal.scrollHeight
  }

  const run = raw => {
    const command = raw.trim()
    write(`diego@luma:${cwd.length ? `~/${cwd.join("/")}` : "~"}$ ${command}`)
    if (!command) return
    if (forbiddenOperators.test(command)) {
      write("Protected mode: shell operators and redirects are not allowed.")
      return
    }
    const parts = command.match(/"[^"]*"|'[^']*'|\S+/g)?.map(part => part.replace(/^["']|["']$/g, "")) || []
    const name = parts.shift()?.toLowerCase()
    if (blockedWords.includes(name)) {
      write(`Protected mode: "${name}" is blocked to keep this system safe.`)
      return
    }
    const current = nodeAt(cwd)
    const commands = {
      help: () => write("Allowed: help, pwd, ls, cd, mkdir, touch, cat, echo, date, whoami, hostname, history, clear, open"),
      pwd: () => write(`/home/diego${cwd.length ? `/${cwd.join("/")}` : ""}`),
      ls: () => write(Object.entries(current?.children || {}).map(([entryName, entry]) => `${entry.type === "folder" ? "[dir]" : "     "} ${entryName}`).join("\n") || "(empty)"),
      cd: () => {
        const target = parts.join(" ")
        if (!target || target === "~" || target === "/") {
          cwd = []
        } else if (target === "..") {
          cwd.pop()
        } else if (current?.children?.[target]?.type === "folder") {
          cwd.push(target)
        } else {
          write(`Folder not found: ${target}`)
        }
      },
      mkdir: () => {
        const folderName = cleanFileName(parts.join(" "))
        if (!folderName) return write("Usage: mkdir folder-name")
        if (current.children[folderName]) return write("That name is already in use.")
        current.children[folderName] = { type: "folder", children: {} }
        saveFs()
        write(`Created folder: ${folderName}`)
      },
      touch: () => {
        let fileName = cleanFileName(parts.join(" "))
        if (!fileName) return write("Usage: touch file-name.txt")
        if (!fileName.includes(".")) fileName += ".txt"
        if (!current.children[fileName]) current.children[fileName] = { type: "file", content: "" }
        saveFs()
        write(`Created file: ${fileName}`)
      },
      cat: () => {
        const fileName = parts.join(" ")
        const file = current?.children?.[fileName]
        write(file?.type === "file" ? file.content : `File not found: ${fileName}`)
      },
      echo: () => write(parts.join(" ")),
      date: () => write(new Date().toString()),
      whoami: () => write("diego"),
      hostname: () => write("luma-web"),
      history: () => write(appState.terminalHistory.map((item, index) => `${index + 1}  ${item}`).join("\n")),
      clear: () => {
        output.textContent = ""
      },
      open: () => {
        const target = parts[0]?.toLowerCase()
        if (appCatalog[target]) {
          openApp(target)
        } else {
          write(`Use an app name: ${Object.keys(appCatalog).join(", ")}`)
        }
      }
    }
    if (commands[name]) commands[name]()
    else write(`Command not found: ${name}. Type "help".`)
  }

  input.addEventListener("keydown", event => {
    if (event.key === "Enter") {
      const command = input.value
      if (command.trim()) appState.terminalHistory.push(command)
      appState.terminalHistoryIndex = appState.terminalHistory.length
      input.value = ""
      run(command)
    }
    if (event.key === "ArrowUp") {
      event.preventDefault()
      appState.terminalHistoryIndex = Math.max(0, appState.terminalHistoryIndex - 1)
      input.value = appState.terminalHistory[appState.terminalHistoryIndex] || ""
    }
    if (event.key === "ArrowDown") {
      event.preventDefault()
      appState.terminalHistoryIndex = Math.min(appState.terminalHistory.length, appState.terminalHistoryIndex + 1)
      input.value = appState.terminalHistory[appState.terminalHistoryIndex] || ""
    }
  })
  terminal.addEventListener("click", () => input.focus())
  setTimeout(() => input.focus(), 0)
}

function openBrowser() {
  const windowEl = createWindow("browser", { title: "Luma Explorer", menu: false, status: "Restricted web frame" })
  const content = windowEl.querySelector(".window-content")
  content.innerHTML = `
    <div class="browser-app">
      <div class="browser-bar">
        <div class="browser-actions">
          <button class="xp-button" data-browser="back" title="Back">←</button>
          <button class="xp-button" data-browser="home" title="Home">⌂</button>
          <button class="xp-button" data-browser="refresh" title="Refresh">↻</button>
        </div>
        <strong>Address</strong>
        <input value="luma://home" aria-label="Web address">
        <button class="xp-button" data-browser="go">Go</button>
      </div>
      <div class="browser-viewport">
        <iframe title="Web page" sandbox="allow-forms allow-scripts allow-same-origin allow-popups"></iframe>
        <div class="browser-home">
          <div class="boot-mark"><span class="boot-orbit"></span><span class="boot-core"></span></div>
          <h1>Luma Explorer</h1>
          <p>A small window to the wide web</p>
          <form class="home-search">
            <input aria-label="Search query" placeholder="Search the web">
            <button class="xp-button">Search</button>
          </form>
        </div>
      </div>
    </div>
  `
  const iframe = content.querySelector("iframe")
  const home = content.querySelector(".browser-home")
  const address = content.querySelector(".browser-bar > input")
  const search = content.querySelector(".home-search input")
  let currentUrl = ""

  const navigate = raw => {
    let value = raw.trim()
    if (!value || value === "luma://home") {
      address.value = "luma://home"
      home.hidden = false
      iframe.hidden = true
      currentUrl = ""
      return
    }
    if (!/^https?:\/\//i.test(value)) {
      if (value.includes(".") && !value.includes(" ")) value = `https://${value}`
      else value = `https://duckduckgo.com/?q=${encodeURIComponent(value)}`
    }
    try {
      const parsed = new URL(value)
      if (!["http:", "https:"].includes(parsed.protocol)) throw new Error()
      home.hidden = true
      iframe.hidden = false
      iframe.src = parsed.href
      address.value = parsed.href
      currentUrl = parsed.href
    } catch {
      toast("Address blocked", "Only regular HTTP and HTTPS addresses are allowed.")
    }
  }

  content.addEventListener("click", event => {
    const action = event.target.closest("[data-browser]")?.dataset.browser
    if (action === "go") navigate(address.value)
    if (action === "home") navigate("luma://home")
    if (action === "refresh" && currentUrl) iframe.src = currentUrl
    if (action === "back") {
      try {
        iframe.contentWindow.history.back()
      } catch {
        navigate("luma://home")
      }
    }
  })
  address.addEventListener("keydown", event => {
    if (event.key === "Enter") navigate(address.value)
  })
  content.querySelector(".home-search").addEventListener("submit", event => {
    event.preventDefault()
    navigate(search.value)
  })
  navigate("luma://home")
}

function openImages() {
  const windowEl = createWindow("images", { menu: false, status: "5 pictures · Stored locally" })
  const content = windowEl.querySelector(".window-content")
  let selected = 1
  content.innerHTML = `
    <div class="media-app">
      <aside class="media-list">
        <h3>My Pictures</h3>
        ${wallpaperNames.map((name, index) => `<button data-picture="${index + 1}" class="${index === 0 ? "active" : ""}"><i class="wall-${index + 1}"></i><span>${name}</span></button>`).join("")}
      </aside>
      <div class="media-main">
        <div class="image-stage"><div class="image-canvas wall-1" role="img" aria-label="${wallpaperNames[0]}"></div></div>
        <div class="media-toolbar">
          <button data-image-action="previous" title="Previous">◀</button>
          <strong data-image-title>${wallpaperNames[0]}</strong>
          <button data-image-action="next" title="Next">▶</button>
          <button data-image-action="wallpaper">Set as wallpaper</button>
        </div>
      </div>
    </div>
  `
  const image = content.querySelector(".image-canvas")
  const title = content.querySelector("[data-image-title]")

  const select = number => {
    selected = ((number - 1 + 5) % 5) + 1
    image.className = `image-canvas wall-${selected}`
    image.setAttribute("aria-label", wallpaperNames[selected - 1])
    title.textContent = wallpaperNames[selected - 1]
    content.querySelectorAll("[data-picture]").forEach(button => button.classList.toggle("active", Number(button.dataset.picture) === selected))
  }

  content.addEventListener("click", event => {
    const picture = event.target.closest("[data-picture]")
    const action = event.target.closest("[data-image-action]")?.dataset.imageAction
    if (picture) select(Number(picture.dataset.picture))
    if (action === "previous") select(selected - 1)
    if (action === "next") select(selected + 1)
    if (action === "wallpaper") setWallpaper(selected)
  })
}

function openVideos() {
  const windowEl = createWindow("videos", { menu: false, status: "2 videos · Generated in your browser" })
  const content = windowEl.querySelector(".window-content")
  content.innerHTML = `
    <div class="media-app">
      <aside class="media-list">
        <h3>My Videos</h3>
        <button data-video="aurora" class="active"><span>🌌</span><span>Aurora Drift</span></button>
        <button data-video="aquarium"><span>🐠</span><span>Pixel Aquarium</span></button>
      </aside>
      <div class="media-main">
        <div class="video-stage"><canvas width="960" height="540"></canvas></div>
        <div class="media-toolbar">
          <button data-video-action="play">❚❚</button>
          <strong data-video-title>Aurora Drift</strong>
          <span data-video-time>00:00</span>
        </div>
      </div>
    </div>
  `
  const canvas = content.querySelector("canvas")
  const ctx = canvas.getContext("2d")
  const title = content.querySelector("[data-video-title]")
  const time = content.querySelector("[data-video-time]")
  const playButton = content.querySelector("[data-video-action]")
  let mode = "aurora"
  let playing = true
  let started = performance.now()

  const renderAurora = elapsed => {
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height)
    gradient.addColorStop(0, "#020720")
    gradient.addColorStop(.65, "#09314c")
    gradient.addColorStop(1, "#041514")
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    for (let i = 0; i < 90; i++) {
      const x = (i * 137) % canvas.width
      const y = (i * 83) % 300
      ctx.fillStyle = `rgba(210,235,255,${.25 + (i % 6) / 10})`
      ctx.fillRect(x, y, i % 4 === 0 ? 2 : 1, i % 4 === 0 ? 2 : 1)
    }
    ctx.globalCompositeOperation = "lighter"
    for (let band = 0; band < 5; band++) {
      ctx.beginPath()
      for (let x = -30; x <= canvas.width + 30; x += 12) {
        const y = 230 + band * 28 + Math.sin(x / 95 + elapsed / 1200 + band) * 75
        if (x === -30) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.strokeStyle = `hsla(${145 + band * 23}, 90%, 62%, .42)`
      ctx.lineWidth = 22 - band * 2
      ctx.stroke()
    }
    ctx.globalCompositeOperation = "source-over"
    ctx.fillStyle = "#061918"
    ctx.beginPath()
    ctx.moveTo(0, 470)
    for (let x = 0; x <= canvas.width; x += 60) ctx.lineTo(x, 430 + Math.sin(x / 90) * 30)
    ctx.lineTo(canvas.width, canvas.height)
    ctx.lineTo(0, canvas.height)
    ctx.fill()
  }

  const renderAquarium = elapsed => {
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height)
    gradient.addColorStop(0, "#0a7caa")
    gradient.addColorStop(1, "#012d55")
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.fillStyle = "rgba(255,255,255,.1)"
    for (let i = 0; i < 10; i++) {
      ctx.beginPath()
      ctx.moveTo(i * 120 - 80, 0)
      ctx.lineTo(i * 120 + 60, canvas.height)
      ctx.lineTo(i * 120 + 130, canvas.height)
      ctx.lineTo(i * 120 + 20, 0)
      ctx.fill()
    }
    for (let i = 0; i < 13; i++) {
      const direction = i % 2 ? -1 : 1
      const travel = (elapsed * (.035 + i * .002) + i * 140) % 1250
      const x = direction > 0 ? travel - 120 : 1080 - travel
      const y = 80 + (i * 61) % 340 + Math.sin(elapsed / 700 + i) * 18
      const color = ["#ffcc35", "#ff6f44", "#7fe6f2", "#b7ef56", "#fa79bb"][i % 5]
      ctx.save()
      ctx.translate(x, y)
      ctx.scale(direction, 1)
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.ellipse(0, 0, 34, 19, 0, 0, Math.PI * 2)
      ctx.fill()
      ctx.beginPath()
      ctx.moveTo(-28, 0)
      ctx.lineTo(-55, -20)
      ctx.lineTo(-55, 20)
      ctx.closePath()
      ctx.fill()
      ctx.fillStyle = "#081c2c"
      ctx.beginPath()
      ctx.arc(17, -5, 3, 0, Math.PI * 2)
      ctx.fill()
      ctx.restore()
    }
    ctx.fillStyle = "#164c39"
    for (let x = 0; x < canvas.width; x += 45) {
      const height = 35 + (x * 17) % 75
      ctx.fillRect(x, canvas.height - height, 8, height)
    }
    ctx.fillStyle = "#c7a36a"
    ctx.fillRect(0, canvas.height - 22, canvas.width, 22)
  }

  const loop = now => {
    const elapsed = now - started
    if (playing) {
      if (mode === "aurora") renderAurora(elapsed)
      else renderAquarium(elapsed)
      time.textContent = new Date(elapsed).toISOString().slice(14, 19)
    }
    const frame = requestAnimationFrame(loop)
    appState.videoLoops.set(windowEl.id, frame)
  }

  content.addEventListener("click", event => {
    const video = event.target.closest("[data-video]")
    const action = event.target.closest("[data-video-action]")?.dataset.videoAction
    if (video) {
      mode = video.dataset.video
      started = performance.now()
      title.textContent = mode === "aurora" ? "Aurora Drift" : "Pixel Aquarium"
      content.querySelectorAll("[data-video]").forEach(button => button.classList.toggle("active", button === video))
    }
    if (action === "play") {
      playing = !playing
      playButton.textContent = playing ? "❚❚" : "▶"
    }
  })
  loop(performance.now())
}

function openSettings() {
  const windowEl = createWindow("settings", { title: "Display Properties", menu: false, status: "Personalization" })
  const content = windowEl.querySelector(".window-content")
  content.innerHTML = `
    <div class="settings-app">
      <aside class="settings-side">
        <h2>Pick your view</h2>
        <p>Choose one of five local pictures or paste a direct image link.</p>
        <p>Your choice is remembered in this browser.</p>
      </aside>
      <section class="settings-main">
        <h2>Desktop background</h2>
        <p>Select a picture for your desktop.</p>
        <div class="wallpaper-grid">
          ${wallpaperNames.map((name, index) => `<button class="wallpaper-choice wall-${index + 1}" data-wallpaper="${index + 1}" title="${name}"></button>`).join("")}
        </div>
        <label for="wallpaper-link"><strong>Use an image link</strong></label>
        <div class="wallpaper-url">
          <input id="wallpaper-link" type="url" placeholder="https://site.test/picture.jpg">
          <button class="xp-button" data-settings="link">Apply link</button>
        </div>
        <p><small>Only HTTP, HTTPS and image data links are accepted.</small></p>
        <button class="xp-button" data-settings="restore">Restore default</button>
      </section>
    </div>
  `
  const linkInput = content.querySelector("#wallpaper-link")
  linkInput.value = appState.customWallpaper

  const refreshSelection = () => {
    content.querySelectorAll("[data-wallpaper]").forEach(button => button.classList.toggle("selected", !appState.customWallpaper && Number(button.dataset.wallpaper) === appState.currentWallpaper))
  }

  content.addEventListener("click", event => {
    const choice = event.target.closest("[data-wallpaper]")
    const action = event.target.closest("[data-settings]")?.dataset.settings
    if (choice) {
      setWallpaper(Number(choice.dataset.wallpaper))
      refreshSelection()
    }
    if (action === "link") {
      const safeUrl = safeImageUrl(linkInput.value)
      if (!safeUrl) {
        toast("Wallpaper link blocked", "Use a valid HTTP, HTTPS or image data link.")
        return
      }
      appState.customWallpaper = safeUrl
      desktop.className = "custom-wallpaper"
      desktop.style.backgroundImage = `url("${safeUrl}")`
      storeSettings()
      refreshSelection()
      toast("Wallpaper changed", "The linked image is now your background.")
    }
    if (action === "restore") {
      setWallpaper(1)
      linkInput.value = ""
      refreshSelection()
    }
  })
  refreshSelection()
}

function setWallpaper(number) {
  appState.currentWallpaper = Math.max(1, Math.min(5, Number(number) || 1))
  appState.customWallpaper = ""
  desktop.className = `wallpaper-${appState.currentWallpaper}`
  desktop.style.backgroundImage = ""
  storeSettings()
  toast("Wallpaper changed", wallpaperNames[appState.currentWallpaper - 1])
}

function storeSettings() {
  try {
    localStorage.setItem("lumaos-settings", JSON.stringify({
      wallpaper: appState.currentWallpaper,
      customWallpaper: appState.customWallpaper
    }))
  } catch {
    toast("Settings unavailable", "This choice will last until the page closes.")
  }
}

function loadSettings() {
  const settings = loadJson("lumaos-settings", { wallpaper: 1, customWallpaper: "" })
  appState.currentWallpaper = Math.max(1, Math.min(5, Number(settings.wallpaper) || 1))
  appState.customWallpaper = safeImageUrl(settings.customWallpaper || "")
  if (appState.customWallpaper) {
    desktop.className = "custom-wallpaper"
    desktop.style.backgroundImage = `url("${appState.customWallpaper}")`
  } else {
    desktop.className = `wallpaper-${appState.currentWallpaper}`
  }
}

function openClock() {
  const windowEl = createWindow("clock", { menu: false, status: "Local system time", width: 390, height: 430 })
  const content = windowEl.querySelector(".window-content")
  content.innerHTML = `
    <div class="clock-app">
      <div class="analog-clock">
        ${Array.from({ length: 12 }, (_, index) => `<div class="clock-number" style="--n:${index + 1}"><span>${index + 1}</span></div>`).join("")}
        <i class="clock-hand hour"></i>
        <i class="clock-hand minute"></i>
        <i class="clock-hand second"></i>
        <i class="clock-pin"></i>
      </div>
      <div class="clock-readout"><strong></strong><span></span></div>
    </div>
  `
  const hour = content.querySelector(".hour")
  const minute = content.querySelector(".minute")
  const second = content.querySelector(".second")
  const readout = content.querySelector(".clock-readout strong")
  const date = content.querySelector(".clock-readout span")

  const render = () => {
    if (!windowEl.isConnected) return
    const now = new Date()
    hour.style.transform = `translateX(-50%) rotate(${(now.getHours() % 12) * 30 + now.getMinutes() / 2}deg)`
    minute.style.transform = `translateX(-50%) rotate(${now.getMinutes() * 6 + now.getSeconds() / 10}deg)`
    second.style.transform = `translateX(-50%) rotate(${now.getSeconds() * 6}deg)`
    readout.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    date.textContent = now.toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" })
    setTimeout(render, 1000)
  }
  render()
}

function openHelp() {
  const windowEl = createWindow("help", { status: "LumaOS Help Center" })
  windowEl.querySelector(".window-content").innerHTML = `
    <article class="help-app">
      <h1>Welcome to LumaOS</h1>
      <p>Double-click a desktop icon, use the Start menu, or right-click the wallpaper to begin.</p>
      <div class="help-grid">
        <div class="help-card"><strong>Move windows</strong>Drag a blue title bar. Use the three title buttons to minimize, maximize or close.</div>
        <div class="help-card"><strong>Keep your work</strong>Notes, folders and wallpaper settings are saved in your browser.</div>
        <div class="help-card"><strong>Draw freely</strong>Luma Paint includes pencil, brush, eraser, undo, colors and PNG saving.</div>
        <div class="help-card"><strong>Stay protected</strong>Safe Terminal permits a small command set and refuses destructive actions.</div>
        <div class="help-card"><strong>Browse carefully</strong>Luma Explorer runs sites in a restricted frame. Some sites may refuse to appear inside frames.</div>
        <div class="help-card"><strong>Personalize</strong>Open Desktop Properties to choose one of five pictures or add a direct image link.</div>
      </div>
    </article>
  `
}

function openRecycle() {
  const windowEl = createWindow("recycle", { status: "0 objects" })
  windowEl.querySelector(".window-content").innerHTML = `
    <div class="file-grid">
      <div class="help-card"><strong>The Recycle Bin is empty</strong>LumaOS never removes your files without a clear request.</div>
    </div>
  `
}

function toast(title, message) {
  const item = document.createElement("div")
  item.className = "toast"
  item.innerHTML = `<strong>${escapeHtml(title)}</strong>${escapeHtml(message)}`
  document.querySelector("#toast-region").append(item)
  setTimeout(() => item.remove(), 3600)
}

function updateClock() {
  const now = new Date()
  document.querySelector("#clock-time").textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  document.querySelector("#clock-date").textContent = now.toLocaleDateString([], { month: "short", day: "numeric" })
}

function closeStartMenu() {
  startMenu.hidden = true
  startButton.setAttribute("aria-expanded", "false")
}

function toggleStartMenu() {
  const opening = startMenu.hidden
  startMenu.hidden = !opening
  startButton.setAttribute("aria-expanded", String(opening))
  contextMenu.hidden = true
}

function showShutdown() {
  closeStartMenu()
  shutdownScreen.hidden = false
}

function bootAgain() {
  shutdownScreen.hidden = true
  desktop.classList.remove("powered-off")
  const boot = document.querySelector("#boot-screen")
  boot.classList.remove("done")
  setTimeout(() => boot.classList.add("done"), 1900)
}

desktopIcons.addEventListener("click", event => {
  const icon = event.target.closest(".desktop-icon")
  desktopIcons.querySelectorAll(".desktop-icon").forEach(item => item.classList.toggle("selected", item === icon))
})

desktopIcons.addEventListener("dblclick", event => {
  const icon = event.target.closest("[data-app]")
  if (icon) openApp(icon.dataset.app)
})

document.addEventListener("click", event => {
  const open = event.target.closest("[data-open]")
  if (open) openApp(open.dataset.open)
  if (!event.target.closest("#start-menu") && !event.target.closest("#start-button")) closeStartMenu()
  if (!event.target.closest("#context-menu")) contextMenu.hidden = true
})

startButton.addEventListener("click", event => {
  event.stopPropagation()
  toggleStartMenu()
})

desktop.addEventListener("contextmenu", event => {
  if (event.target.closest(".os-window") || event.target.closest(".taskbar") || event.target.closest(".start-menu")) return
  event.preventDefault()
  contextMenu.hidden = false
  contextMenu.style.left = `${Math.min(event.clientX, window.innerWidth - 185)}px`
  contextMenu.style.top = `${Math.min(event.clientY, window.innerHeight - 205)}px`
  closeStartMenu()
})

contextMenu.addEventListener("click", event => {
  const action = event.target.closest("[data-context]")?.dataset.context
  if (action === "new-folder") {
    openFiles()
    setTimeout(() => document.querySelector(".os-window:last-child [data-file-action='folder']")?.click(), 100)
  }
  if (action === "notepad") openNotepad()
  if (action === "wallpaper") openSettings()
  if (action === "refresh") toast("Desktop refreshed", "Everything is in its place.")
})

document.querySelector("#log-off").addEventListener("click", () => {
  closeStartMenu()
  toast("Session locked", "Your local work remains available.")
  const boot = document.querySelector("#boot-screen")
  boot.classList.remove("done")
  setTimeout(() => boot.classList.add("done"), 1300)
})

document.querySelector("#power-off").addEventListener("click", showShutdown)
document.querySelector("#cancel-power").addEventListener("click", () => {
  shutdownScreen.hidden = true
})

shutdownScreen.addEventListener("click", event => {
  const action = event.target.closest("[data-power]")?.dataset.power
  if (action === "standby") {
    shutdownScreen.hidden = true
    desktop.classList.add("powered-off")
    setTimeout(() => {
      desktop.classList.remove("powered-off")
      toast("Welcome back", "LumaOS resumed safely.")
    }, 1800)
  }
  if (action === "off") {
    shutdownScreen.hidden = true
    desktop.classList.add("powered-off")
    setTimeout(() => {
      const boot = document.querySelector("#boot-screen")
      boot.classList.remove("done")
      boot.querySelector(".boot-copy").textContent = "Click anywhere to start"
      boot.addEventListener("click", () => {
        boot.querySelector(".boot-copy").textContent = "A brighter personal web"
        bootAgain()
      }, { once: true })
    }, 900)
  }
  if (action === "restart") bootAgain()
})

document.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    closeStartMenu()
    contextMenu.hidden = true
    shutdownScreen.hidden = true
  }
})

window.addEventListener("resize", () => {
  document.querySelectorAll(".os-window:not(.maximized)").forEach(windowEl => {
    windowEl.style.left = `${Math.max(0, Math.min(windowEl.offsetLeft, window.innerWidth - 100))}px`
    windowEl.style.top = `${Math.max(0, Math.min(windowEl.offsetTop, window.innerHeight - 90))}px`
  })
})

renderDesktopIcons()
renderStartApps()
loadSettings()
updateClock()
setInterval(updateClock, 1000)
setTimeout(() => {
  document.querySelector("#boot-screen").classList.add("done")
  toast("Welcome to LumaOS", "Double-click an icon or press start.")
}, 2100)
