const parentOrigin = window.location.origin

const notify = (type, detail = '') => {
  window.parent.postMessage({ source: 'superblue-player', type, detail }, parentOrigin)
}

const loadGame = file => {
  const readyTimer = window.setTimeout(() => notify('error', 'The emulator core is taking too long to load. Check your internet connection and try again.'), 20000)
  window.EJS_player = '#game'
  window.EJS_core = 'snes'
  window.EJS_gameUrl = file
  window.EJS_gameName = file.name.replace(/\.[^.]+$/, '')
  window.EJS_pathtodata = 'https://cdn.emulatorjs.org/4.2.3/data/'
  window.EJS_startOnLoaded = false
  window.EJS_color = '#1261d8'
  window.EJS_backgroundColor = '#07101f'
  window.EJS_disableAutoLang = true
  window.EJS_forceLegacyCores = true
  window.EJS_defaultOptions = { shader: 'disabled' }
  window.EJS_mouse = false
  window.EJS_multitap = false
  window.EJS_Buttons = { fullscreen: true }
  window.EJS_ready = () => {
    window.clearTimeout(readyTimer)
    notify('ready')
  }
  window.EJS_onGameStart = () => notify('started')

  const loader = document.createElement('script')
  loader.src = `${window.EJS_pathtodata}loader.js`
  loader.onerror = () => {
    window.clearTimeout(readyTimer)
    notify('error', 'The emulator core could not be loaded. Check your internet connection.')
  }
  document.body.append(loader)
}

window.addEventListener('message', event => {
  if (event.origin !== parentOrigin || event.data?.type !== 'load-rom' || !(event.data.file instanceof File)) return
  loadGame(event.data.file)
}, { once: true })

window.addEventListener('error', event => notify('error', event.message || 'The player stopped unexpectedly.'))
window.addEventListener('unhandledrejection', () => notify('error', 'The ROM could not be started.'))
window.addEventListener('keydown', event => {
  if (event.key === 'Escape') notify('exit-fullscreen')
})
notify('waiting')
