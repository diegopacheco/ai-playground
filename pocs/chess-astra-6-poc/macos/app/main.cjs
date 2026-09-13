const { app, BrowserWindow, Menu, ipcMain } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const services = require('./services.cjs');
const windowState = require('./windowState.cjs');
const shortcuts = require('./shortcuts.json');

const rooms = [['library', 'The Library'], ['greatHall', 'The Great Hall'], ['office', 'Dumbledore’s Office']];
let win = null;
let state = null;
let config = null;
let drag = null;
let stopping = false;
let stopped = false;

if (process.env.WIZARDS_GAMBIT_USER_DATA) app.setPath('userData', process.env.WIZARDS_GAMBIT_USER_DATA);
if (!app.requestSingleInstanceLock()) app.exit(0);

const send = (channel, payload) => { if (win && !win.isDestroyed()) win.webContents.send(channel, payload); };
const command = (type, value) => send('command', { type, value });

function injectOverlay() {
  win.webContents.insertCSS(fs.readFileSync(path.join(__dirname, 'overlay.css'), 'utf8'));
  win.webContents.executeJavaScript(`window.SHORTCUTS = ${JSON.stringify(shortcuts)};\n${fs.readFileSync(path.join(__dirname, 'overlay.js'), 'utf8')}`);
}

async function boot() {
  await win.loadFile(path.join(__dirname, 'boot.html'));
  try {
    config = services.readConfig();
  } catch (error) {
    send('boot', { id: 'runtime', status: 'failed', detail: `Missing app configuration. Reinstall with scripts/install-macos.sh (${error.message})` });
    return;
  }
  const ready = await services.startAll(config, (id, status, detail) => send('boot', { id, status, detail }));
  if (ready && win && !win.isDestroyed()) win.loadURL(config.url);
}

function capture() {
  const file = path.join(app.getPath('desktop'), `Wizards Gambit ${new Date().toISOString().replace(/[:.]/g, '-')}.png`);
  spawn('/usr/sbin/screencapture', ['-i', file]).on('close', () => command('toast', fs.existsSync(file) ? `Saved “${path.basename(file)}” to the Desktop` : 'Capture cancelled'));
}

function menu() {
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: app.name, submenu: [{ role: 'about' }, { type: 'separator' }, { role: 'hide' }, { role: 'hideOthers' }, { role: 'unhide' }, { type: 'separator' }, { id: 'quit', label: 'Quit and Stop Game Server', accelerator: 'CmdOrCtrl+Q', click: () => app.quit() }] },
    { label: 'Edit', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    {
      label: 'Go',
      submenu: [
        { id: 'search', label: 'Search…', accelerator: 'CmdOrCtrl+K', click: () => command('search') },
        { id: 'shortcuts', label: 'Keyboard Shortcuts', accelerator: 'CmdOrCtrl+/', click: () => command('shortcuts') },
        { type: 'separator' },
        ...rooms.map(([id, label], index) => ({ id: `room-${id}`, label, accelerator: `CmdOrCtrl+${index + 1}`, click: () => command('room', id) })),
      ],
    },
    {
      label: 'View',
      submenu: [
        { role: 'zoomIn' },
        { role: 'zoomIn', accelerator: 'CmdOrCtrl+=', visible: false, acceleratorWorksWhenHidden: true },
        { role: 'zoomOut' },
        { role: 'resetZoom' },
        { type: 'separator' },
        { id: 'fullscreen', label: 'Toggle Full Screen', accelerator: 'CmdOrCtrl+Shift+Enter', click: () => win?.setFullScreen(!win.isFullScreen()) },
        { id: 'capture', label: 'Capture Screen Area', accelerator: 'CmdOrCtrl+P', click: capture },
      ],
    },
    { role: 'windowMenu' },
  ]));
}

function createWindow() {
  state = windowState.load();
  win = new BrowserWindow({
    ...state.bounds,
    minWidth: 480,
    minHeight: 600,
    show: false,
    title: 'Wizards Gambit',
    backgroundColor: '#f7f5ef',
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 14, y: 12 },
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, sandbox: true },
  });
  windowState.track(win, state);
  win.once('ready-to-show', () => {
    win.show();
    if (state.fullscreen) win.setFullScreen(true);
  });
  win.webContents.on('dom-ready', injectOverlay);
  win.on('closed', () => { win = null; });
  boot();
}

ipcMain.on('retry', () => boot());
ipcMain.on('quit', () => app.quit());
ipcMain.on('drag-start', (_, x, y) => {
  const [left, top] = win.getPosition();
  drag = { x, y, left, top };
});
ipcMain.on('drag-move', (_, x, y) => {
  if (drag && !win.isFullScreen()) win.setPosition(Math.round(drag.left + x - drag.x), Math.round(drag.top + y - drag.y));
});
ipcMain.on('toggle-maximize', () => {
  if (!win.isFullScreen()) windowState.toggleMaximize(win, state);
});

app.on('second-instance', () => {
  if (!win) return;
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', async event => {
  if (stopped) return;
  event.preventDefault();
  if (stopping) return;
  stopping = true;
  command('toast', 'Stopping the game server…');
  try {
    await services.stopAll(config ?? services.readConfig());
  } catch (error) {
    console.error(error);
  }
  stopped = true;
  app.quit();
});
for (const signal of ['SIGTERM', 'SIGINT']) process.on(signal, () => app.quit());
app.whenReady().then(() => {
  menu();
  createWindow();
});
