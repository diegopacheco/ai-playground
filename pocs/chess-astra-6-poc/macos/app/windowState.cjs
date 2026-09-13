const { app, screen } = require('electron');
const fs = require('node:fs');
const path = require('node:path');

const file = () => path.join(app.getPath('userData'), 'window-state.json');
const fallback = { bounds: { width: 1440, height: 960 }, fullscreen: false, restore: null };

function visible(bounds) {
  return screen.getAllDisplays().some(({ workArea }) => bounds.x < workArea.x + workArea.width && bounds.x + bounds.width > workArea.x && bounds.y < workArea.y + workArea.height && bounds.y + bounds.height > workArea.y);
}

function load() {
  try {
    const state = JSON.parse(fs.readFileSync(file(), 'utf8'));
    if (Number.isFinite(state.bounds?.x) && visible(state.bounds)) return state;
  } catch {}
  return { ...fallback, bounds: { ...fallback.bounds } };
}

function track(win, state) {
  const save = () => {
    if (win.isDestroyed()) return;
    const fullscreen = win.isFullScreen();
    const bounds = fullscreen ? state.bounds : win.getBounds();
    Object.assign(state, { bounds, fullscreen });
    fs.mkdirSync(path.dirname(file()), { recursive: true });
    fs.writeFileSync(file(), JSON.stringify(state));
  };
  let timer;
  const later = () => { clearTimeout(timer); timer = setTimeout(save, 250); };
  for (const event of ['move', 'resize', 'enter-full-screen', 'leave-full-screen']) win.on(event, later);
  win.on('close', () => { clearTimeout(timer); save(); });
}

function toggleMaximize(win, state) {
  const { workArea } = screen.getDisplayMatching(win.getBounds());
  if (state.restore) {
    win.setBounds(state.restore);
    state.restore = null;
  } else {
    state.restore = win.getBounds();
    win.setBounds(workArea);
  }
}

module.exports = { load, track, toggleMaximize };
