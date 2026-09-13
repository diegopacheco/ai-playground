const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('macApp', {
  onBoot: callback => ipcRenderer.on('boot', (_, step) => callback(step)),
  onCommand: callback => ipcRenderer.on('command', (_, command) => callback(command)),
  retry: () => ipcRenderer.send('retry'),
  quit: () => ipcRenderer.send('quit'),
  dragStart: (x, y) => ipcRenderer.send('drag-start', x, y),
  dragMove: (x, y) => ipcRenderer.send('drag-move', x, y),
  toggleMaximize: () => ipcRenderer.send('toggle-maximize'),
});
