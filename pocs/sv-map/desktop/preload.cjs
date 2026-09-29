const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("svmap", {
  toggleMaximize: () => ipcRenderer.send("toggle-maximize"),
  onToast: callback => ipcRenderer.on("toast", (_, text) => callback(text)),
  onBoot: callback => ipcRenderer.on("boot", (_, step) => callback(step))
});
