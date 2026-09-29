const { contextBridge, ipcRenderer } = require("electron")

contextBridge.exposeInMainWorld("moviesMap", {
  toggleMaximize: () => ipcRenderer.send("toggle-maximize"),
  retryBoot: () => ipcRenderer.send("retry-boot"),
  onToast: fn => ipcRenderer.on("toast", (_, text) => fn(text)),
  onBootStep: fn => ipcRenderer.on("boot-step", (_, step) => fn(step))
})
