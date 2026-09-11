import { PatchEngine } from './patch-engine.js'

const origin = location.origin
const session = Number(new URL(location.href).searchParams.get('session'))
let loaded = false
let engine
let smooth = false
let fpsTimer
let lastFrame = 0
const notify = (type, detail = {}) => parent.postMessage({ source: 'emu-magic-player', session, type, ...detail }, origin)

function loadGame(file) {
  loaded = true
  const timer = setTimeout(() => notify('error', { message: 'Core loading is taking longer than expected. Check access to cdn.emulatorjs.org or reload the cartridge.' }), 45000)
  window.EJS_player = '#game'
  window.EJS_core = 'snes'
  window.EJS_gameUrl = file
  window.EJS_gameName = file.name.replace(/\.[^.]+$/, '')
  window.EJS_pathtodata = 'https://cdn.emulatorjs.org/4.2.3/data/'
  window.EJS_startOnLoaded = false
  window.EJS_color = '#294d36'
  window.EJS_backgroundColor = '#111910'
  window.EJS_disableAutoLang = true
  window.EJS_forceLegacyCores = false
  window.EJS_threads = false
  window.EJS_defaultOptions = { shader: 'disabled', vsync: 'disabled', slowMotion: 'disabled', fastForward: 'disabled', rewindEnabled: 'disabled' }
  window.EJS_noAutoFocus = false
  window.EJS_mouse = false
  window.EJS_multitap = false
  window.EJS_Buttons = { cheat: false, saveState: false, loadState: false, quickSave: false, quickLoad: false, restart: false, fullscreen: false }
  window.EJS_ready = () => {
    clearTimeout(timer)
    notify('ready')
  }
  window.EJS_onGameStart = () => {
    window.EJS_emulator.changeSettingOption('vsync', 'disabled')
    window.EJS_emulator.changeSettingOption('slowMotion', 'disabled')
    window.EJS_emulator.changeSettingOption('fastForward', 'disabled')
    window.EJS_emulator.cheats = []
    window.EJS_emulator.gameManager.resetCheat()
    engine = new PatchEngine(window.EJS_emulator, settleChanges, pauseEmulator)
    setFilter(smooth ? 'on' : 'off')
    window.EJS_emulator.elements?.parent?.focus()
    lastFrame = window.EJS_emulator.gameManager.getFrameNum()
    let lastTime = performance.now()
    clearInterval(fpsTimer)
    fpsTimer = setInterval(() => {
      const frame = window.EJS_emulator.gameManager.getFrameNum()
      const now = performance.now()
      notify('telemetry', { fps: Math.max(0, Math.min(120, Math.round((frame - lastFrame) * 1000 / Math.max(1, now - lastTime)))), threaded: false, webgl2: window.EJS_emulator.webgl2Enabled !== false })
      lastFrame = frame
      lastTime = now
    }, 1000)
    notify('started')
  }
  const loader = document.createElement('script')
  loader.src = `${window.EJS_pathtodata}loader.js`
  loader.onerror = () => {
    clearTimeout(timer)
    notify('error', { message: 'Could not download the emulator core. Check access to cdn.emulatorjs.org.' })
  }
  document.body.append(loader)
}

async function pauseEmulator() {
  const emulator = window.EJS_emulator
  emulator.pause()
  let frame = emulator.gameManager.getFrameNum()
  let stableSince = performance.now()
  const deadline = stableSince + 2000
  do {
    await new Promise(resolve => setTimeout(resolve, 16))
    const current = emulator.gameManager.getFrameNum()
    if (current !== frame) {
      frame = current
      stableSince = performance.now()
    }
    if (performance.now() > deadline) throw new Error('The emulator did not finish pausing for the checkpoint.')
  } while (performance.now() - stableSince < 50)
}

async function settleChanges() {
  const emulator = window.EJS_emulator
  const paused = emulator.paused
  const target = emulator.gameManager.getFrameNum() + 3
  const deadline = performance.now() + 2000
  emulator.play()
  try {
    while (emulator.gameManager.getFrameNum() < target) {
      if (performance.now() > deadline) throw new Error('The emulator did not advance to verify the change.')
      await new Promise(resolve => setTimeout(resolve, 16))
    }
  } finally {
    if (paused) await pauseEmulator()
  }
}

function setFilter(value) {
  smooth = value === 'on'
  if (window.EJS_emulator?.canvas) {
    window.EJS_emulator.canvas.style.imageRendering = smooth ? 'auto' : 'pixelated'
    window.EJS_emulator.canvas.style.filter = smooth ? 'saturate(1.14) contrast(1.06)' : 'none'
  }
  notify('filter', { smooth })
}

function forwardKey(data) {
  const target = window.EJS_emulator?.elements?.parent
  if (!target) return
  const event = new KeyboardEvent(data.action, { key: data.key, code: data.code, bubbles: true, cancelable: true })
  Object.defineProperty(event, 'keyCode', { value: data.keyCode })
  target.dispatchEvent(event)
}

window.addEventListener('pointerdown', () => window.EJS_emulator?.elements?.parent?.focus())
window.addEventListener('keydown', event => {
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(event.key) && !event.target.closest('input, textarea, select, [contenteditable=true]')) event.preventDefault()
}, { capture: true })

window.addEventListener('message', async event => {
  if (event.origin !== origin || event.source !== parent) return
  const data = event.data
  if (data?.type === 'load-rom' && !loaded && data.file instanceof File) loadGame(data.file)
  if (data?.type === 'filter') setFilter(data.value)
  if (data?.type === 'key') forwardKey(data)
  if (!['apply', 'undo'].includes(data?.type)) return
  try {
    if (!engine) throw new Error('Start the cartridge before changing the game.')
    const result = data.type === 'apply' ? await engine.apply(data.cheats) : await engine.undo()
    notify('result', { id: data.id, ...result })
  } catch (error) {
    notify('result', { id: data.id, error: error.message })
  }
})
function reportError(message, event) {
  if (/wake lock/i.test(message)) {
    event.preventDefault()
    notify('warning', { message: 'The browser blocked keeping the screen awake.' })
  } else notify('error', { message })
}
window.addEventListener('error', event => reportError(event.message || 'The emulator stopped unexpectedly.', event))
window.addEventListener('unhandledrejection', event => reportError(event.reason?.message || 'The cartridge could not be started.', event))
notify('waiting')
