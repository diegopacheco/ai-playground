const supportedExtensions = new Set(['sfc', 'smc', 'fig', 'swc', 'gd3', 'gd7', 'dx2', 'bsx', 'zip', '7z'])
const maximumRomSize = 64 * 1024 * 1024

const elements = {
  dropZone: document.querySelector('#dropZone'),
  romInput: document.querySelector('#romInput'),
  loadedCart: document.querySelector('#loadedCart'),
  changeRom: document.querySelector('#changeRom'),
  romName: document.querySelector('#romName'),
  romSize: document.querySelector('#romSize'),
  emptyState: document.querySelector('#emptyState'),
  loadingState: document.querySelector('#loadingState'),
  loadingName: document.querySelector('#loadingName'),
  playerFrame: document.querySelector('#playerFrame'),
  screenShell: document.querySelector('#screenShell'),
  fullscreenButton: document.querySelector('#fullscreenButton'),
  fullscreenLabel: document.querySelector('#fullscreenLabel'),
  statusText: document.querySelector('#statusText'),
  statusPill: document.querySelector('#statusPill'),
  toast: document.querySelector('#toast')
}

let selectedRom = null
let toastTimer = null

const extensionOf = file => file.name.split('.').pop()?.toLowerCase() || ''

const formatBytes = bytes => {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

const setStatus = (text, state = '') => {
  elements.statusText.textContent = text
  elements.statusPill.dataset.state = state
}

const showToast = message => {
  window.clearTimeout(toastTimer)
  elements.toast.textContent = message
  elements.toast.classList.add('visible')
  toastTimer = window.setTimeout(() => elements.toast.classList.remove('visible'), 3600)
}

const validateRom = file => {
  if (!file) return 'Choose a ROM file first.'
  if (!supportedExtensions.has(extensionOf(file))) return 'That file type is not supported. Try an SFC, SMC, FIG, SWC, ZIP, or 7Z file.'
  if (file.size === 0) return 'That ROM file is empty.'
  if (file.size > maximumRomSize) return 'That file is larger than the 64 MB safety limit.'
  return ''
}

const showLoading = file => {
  elements.emptyState.hidden = true
  elements.playerFrame.hidden = true
  elements.loadingState.hidden = false
  elements.loadingName.textContent = file.name
  elements.dropZone.hidden = true
  elements.loadedCart.hidden = false
  elements.romName.textContent = file.name
  elements.romSize.textContent = `${extensionOf(file).toUpperCase()} · ${formatBytes(file.size)}`
  elements.fullscreenButton.disabled = false
  setStatus('Reading cartridge', 'loading')
}

const mountPlayer = file => {
  elements.playerFrame.onload = () => {
    elements.playerFrame.contentWindow.postMessage({ type: 'load-rom', file }, window.location.origin)
  }
  elements.playerFrame.src = `/player.html?session=${Date.now()}`
}

const loadRom = file => {
  const error = validateRom(file)
  if (error) {
    showToast(error)
    return
  }
  selectedRom = file
  showLoading(file)
  mountPlayer(file)
}

const resetPlayer = () => {
  selectedRom = null
  elements.playerFrame.onload = null
  elements.playerFrame.src = 'about:blank'
  elements.playerFrame.hidden = true
  elements.loadingState.hidden = true
  elements.emptyState.hidden = false
  elements.dropZone.hidden = false
  elements.loadedCart.hidden = true
  elements.fullscreenButton.disabled = true
  elements.romInput.value = ''
  setStatus('Ready for a cartridge')
}

elements.dropZone.addEventListener('click', () => elements.romInput.click())
elements.changeRom.addEventListener('click', resetPlayer)
elements.romInput.addEventListener('change', event => loadRom(event.target.files[0]))

for (const eventName of ['dragenter', 'dragover']) {
  window.addEventListener(eventName, event => {
    event.preventDefault()
    elements.dropZone.classList.add('dragging')
  })
}

for (const eventName of ['dragleave', 'drop']) {
  window.addEventListener(eventName, event => {
    event.preventDefault()
    elements.dropZone.classList.remove('dragging')
  })
}

window.addEventListener('drop', event => loadRom(event.dataTransfer.files[0]))

window.addEventListener('message', event => {
  if (event.origin !== window.location.origin || event.data?.source !== 'superblue-player') return
  if (event.data.type === 'ready') {
    elements.loadingState.hidden = true
    elements.playerFrame.hidden = false
    elements.fullscreenButton.disabled = false
    setStatus('Cartridge ready', 'ready')
  }
  if (event.data.type === 'waiting') {
    elements.loadingState.hidden = true
    elements.playerFrame.hidden = false
    setStatus('Loading emulator core', 'loading')
  }
  if (event.data.type === 'started') setStatus('Now playing', 'playing')
  if (event.data.type === 'exit-fullscreen' && document.fullscreenElement) document.exitFullscreen()
  if (event.data.type === 'error') {
    setStatus('Cartridge error', 'error')
    showToast(event.data.detail || 'This ROM could not be loaded.')
  }
})

elements.fullscreenButton.addEventListener('click', async () => {
  if (!selectedRom) return
  try {
    if (document.fullscreenElement) await document.exitFullscreen()
    else await elements.screenShell.requestFullscreen()
  } catch {
    showToast('Fullscreen is not available in this browser.')
  }
})

document.addEventListener('fullscreenchange', () => {
  const isFullscreen = Boolean(document.fullscreenElement)
  elements.fullscreenLabel.textContent = isFullscreen ? 'EXIT SCREEN' : 'FULLSCREEN'
  if (isFullscreen) setStatus('Fullscreen · ESC to exit', 'playing')
  else if (selectedRom) setStatus('Cartridge ready', 'ready')
})

window.addEventListener('keydown', event => {
  if (event.key === 'Escape' && document.fullscreenElement) document.exitFullscreen()
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'o') {
    event.preventDefault()
    elements.romInput.click()
  }
})
