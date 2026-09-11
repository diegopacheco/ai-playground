import { validateReply } from './contracts.js'

const $ = selector => document.querySelector(selector)
const frame = $('#player')
let file
let game
let started = false
let available = false
let busy = false
let cheats = []
let checkpoints = 0
let generation = 0
let controller
let noticeTimer
const pending = new Map()
const gameKeyCodes = { ArrowUp: 38, ArrowDown: 40, ArrowLeft: 37, ArrowRight: 39, Enter: 13, z: 90, x: 88, a: 65, s: 83 }

function notice(message) {
  clearTimeout(noticeTimer)
  $('#notice').textContent = message
  $('#notice').hidden = false
  noticeTimer = setTimeout(() => { $('#notice').hidden = true }, 8000)
}

function controls() {
  $('#send').disabled = !started || !available || busy
  $('#undo').disabled = !started || busy || !checkpoints
  $('#filterToggle').disabled = !started
  $('#cancel').hidden = !controller
  $('#requestStatus').textContent = busy ? 'Working on your change…' : !started ? 'Load and start a game' : !available ? 'Codex unavailable' : `${$('#model').value} · ready`
}

function message(author, text, detail = '') {
  const article = document.createElement('article')
  article.className = `message ${author.toLowerCase()}`
  const label = document.createElement('span')
  label.className = 'message-author'
  label.textContent = author === 'user' ? 'YOU' : author.toUpperCase()
  const paragraph = document.createElement('p')
  paragraph.textContent = text
  article.append(label, paragraph)
  if (detail) {
    const note = document.createElement('small')
    note.textContent = detail
    article.append(note)
  }
  $('#conversation').append(article)
  $('#conversation').scrollTop = $('#conversation').scrollHeight
}

function renderChanges(result) {
  cheats = result.cheats
  checkpoints = result.checkpoints
  $('#changeCount').textContent = String(cheats.length).padStart(2, '0')
  $('#activeChanges').replaceChildren()
  if (!cheats.length) {
    const empty = document.createElement('p')
    empty.className = 'no-changes'
    empty.textContent = 'No active memory codes.'
    $('#activeChanges').append(empty)
  }
  for (const cheat of cheats.slice(0, 3)) {
    const card = document.createElement('div')
    card.className = 'change-card'
    const label = document.createElement('strong')
    label.textContent = cheat.label
    const code = document.createElement('code')
    code.textContent = cheat.code
    card.append(label, code)
    $('#activeChanges').append(card)
  }
  $('#checkpointNote').textContent = `${checkpoints}/5 checkpoints available. Undo restores gameplay to before the last change. Applied codes do not guarantee the intended effect.`
  controls()
}

function command(type, values = {}) {
  return new Promise((resolve, reject) => {
    const id = crypto.randomUUID()
    const timer = setTimeout(() => {
      pending.delete(id)
      started = false
      controls()
      reject(new Error('The player did not acknowledge the operation. Reload the cartridge before sending more changes.'))
    }, 10000)
    pending.set(id, { resolve, reject, timer })
    frame.contentWindow.postMessage({ type, id, ...values }, location.origin)
  })
}

async function loadRom(selected) {
  if (!selected) return
  if (!/\.(sfc|smc)$/i.test(selected.name) || !selected.size || selected.size > 64 * 1024 * 1024) return notice('Choose a non-empty .sfc or .smc cartridge under 64 MB. Extract archives first.')
  const session = ++generation
  controller?.abort()
  controller = null
  busy = false
  started = false
  frame.src = 'about:blank'
  frame.classList.remove('smooth')
  frame.hidden = true
  $('#emptyScreen').hidden = false
  $('#fullscreen').disabled = true
  $('#filterToggle').disabled = true
  $('#filterToggle').textContent = 'HD filter off'
  game = null
  for (const task of pending.values()) {
    clearTimeout(task.timer)
    task.reject(new Error('Cartridge changed.'))
  }
  pending.clear()
  controls()
  file = selected
  $('#gameName').textContent = file.name
  $('#gameDetails').textContent = `${(file.size / 1024).toFixed(0)} KB · ROM stays in your browser`
  $('#gameStatus').textContent = 'READING CARTRIDGE'
  $('#gameLed').classList.remove('on')
  $('#screenNote').textContent = 'SNES / SNES9X'
  renderChanges({ cheats: [], checkpoints: 0 })
  $('#conversation').replaceChildren()
  message('codex', 'Cartridge loaded. Start the game, then tell me what you want to change.', 'The game title, SHA-256, your request, and active codes are sent to Codex. ROM bytes and save states stay here.')
  try {
    const bytes = new Uint8Array(await selected.arrayBuffer())
    const digest = await crypto.subtle.digest('SHA-256', bytes)
    if (session !== generation) return
    const offset = bytes.length % 1024 === 512 ? 512 : 0
    const titles = [0x7fc0, 0xffc0, 0x40ffc0].map(address => new TextDecoder().decode(bytes.subarray(offset + address, offset + address + 21))).filter(title => /^[\x20-\x7e]{4,21}$/.test(title))
    game = { name: selected.name.slice(0, 200), internalTitle: titles[0]?.trim() || '', sha256: Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join(''), size: selected.size }
    $('#emptyScreen').hidden = true
    frame.hidden = false
    $('#fullscreen').disabled = false
    frame.src = `/player.html?session=${session}`
    $('#gameStatus').textContent = 'LOADING CORE'
  } catch (error) {
    if (session === generation) notice(`Could not read the cartridge: ${error.message}`)
  }
}

window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.source !== frame.contentWindow || event.data?.source !== 'emu-magic-player') return
  const data = event.data
  if (data.session !== generation) return
  if (data.type === 'warning') notice(data.message)
  if (data.type === 'waiting' && file) frame.contentWindow.postMessage({ type: 'load-rom', file }, location.origin)
  if (data.type === 'ready') $('#gameStatus').textContent = 'PRESS START GAME'
  if (data.type === 'started') {
    started = true
    $('#gameStatus').textContent = 'NOW PLAYING'
    $('#gameLed').classList.add('on')
    $('#filterToggle').disabled = false
    controls()
  }
  if (data.type === 'telemetry') $('#screenNote').textContent = `${data.fps} FPS · ${data.threaded ? 'threaded' : 'single-thread'} · ${data.webgl2 ? 'WebGL2' : 'WebGL1'}`
  if (data.type === 'filter') {
    $('#filterToggle').textContent = data.smooth ? 'HD filter on' : 'HD filter off'
    frame.classList.toggle('smooth', data.smooth)
  }
  if (data.type === 'error') {
    started = false
    controller?.abort()
    $('#gameStatus').textContent = 'PLAYER ERROR'
    $('#gameLed').classList.remove('on')
    notice(data.message)
    controls()
  }
  if (data.type === 'result') {
    const task = pending.get(data.id)
    if (!task) return
    clearTimeout(task.timer)
    pending.delete(data.id)
    if (data.error) task.reject(new Error(data.error))
    else task.resolve(data)
  }
})

$('#promptForm').addEventListener('submit', async event => {
  event.preventDefault()
  if (!started || !available || busy || !game) return
  const prompt = $('#prompt').value.trim()
  if (!prompt) return
  const session = generation
  busy = true
  const requestController = new AbortController()
  controller = requestController
  controls()
  message('user', prompt)
  $('#prompt').value = ''
  try {
    const response = await fetch('/api/change', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt, game, cheats, model: $('#model').value }), signal: requestController.signal })
    const body = await response.json()
    if (!response.ok) throw new Error(body.error || 'Codex request failed.')
    const reply = validateReply(body)
    if (session !== generation || requestController.signal.aborted || !started) return
    if (reply.unsupported) message('codex', reply.message, 'No changes applied.')
    else {
      controller = null
      controls()
      const result = await command('apply', { cheats: reply.cheats })
      if (session !== generation) return
      renderChanges(result)
      message('codex', [reply.message, result.verification].filter(Boolean).join('\n\n'), result.verification ? 'Undo restores the checkpoint.' : 'Codes sent to the emulator; their game effect is unverified. Undo restores the checkpoint.')
    }
  } catch (error) {
    if (session !== generation) return
    message(error.name === 'AbortError' ? 'codex' : 'error', error.name === 'AbortError' ? 'Request cancelled. No changes applied.' : error.message)
    $('#prompt').value ||= prompt
  } finally {
    if (session === generation) {
      busy = false
      controller = null
      controls()
    }
  }
})

$('#undo').addEventListener('click', async () => {
  if (busy || !checkpoints) return
  const session = generation
  busy = true
  controls()
  try {
    const result = await command('undo')
    if (session !== generation) return
    renderChanges(result)
    message('codex', 'Restored the previous checkpoint and its active codes.')
  } catch (error) {
    if (session === generation) notice(error.message)
  } finally {
    if (session === generation) { busy = false; controls() }
  }
})
$('#inspectChanges').addEventListener('click', () => message('codex', cheats.length ? cheats.map(cheat => `${cheat.label}: ${cheat.code}`).join('\n') : 'No active memory codes.'))
$('#cancel').addEventListener('click', () => controller?.abort())
$('#loadButton').addEventListener('click', () => $('#romInput').click())
$('#changeRom').addEventListener('click', () => $('#romInput').click())
$('#romInput').addEventListener('change', event => { loadRom(event.target.files[0]); event.target.value = '' })
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => { $('#prompt').value = button.dataset.prompt; $('#prompt').focus() }))
$('#prompt').addEventListener('keydown', event => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) { event.preventDefault(); $('#promptForm').requestSubmit() }
})
$('#fullscreen').addEventListener('click', async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('#screen').requestFullscreen() } catch { notice('Fullscreen is unavailable in this browser.') }
})
$('#filterToggle').addEventListener('click', () => {
  const enabled = !frame.classList.contains('smooth')
  frame.classList.toggle('smooth', enabled)
  frame.contentWindow.postMessage({ type: 'filter', value: enabled ? 'on' : 'off' }, location.origin)
  $('#filterToggle').textContent = enabled ? 'HD filter on' : 'HD filter off'
})
window.addEventListener('keydown', event => {
  const keyCode = gameKeyCodes[event.key]
  if (!started || !keyCode || (event.target.closest('input, textarea, select, [contenteditable=true]') || event.metaKey || event.ctrlKey || event.altKey)) return
  event.preventDefault()
  frame.contentWindow.postMessage({ type: 'key', action: 'keydown', key: event.key, code: event.code, keyCode }, location.origin)
})
window.addEventListener('keyup', event => {
  const keyCode = gameKeyCodes[event.key]
  if (!started || !keyCode || (event.target.closest('input, textarea, select, [contenteditable=true]') || event.metaKey || event.ctrlKey || event.altKey)) return
  event.preventDefault()
  frame.contentWindow.postMessage({ type: 'key', action: 'keyup', key: event.key, code: event.code, keyCode }, location.origin)
})
for (const name of ['dragenter', 'dragover']) window.addEventListener(name, event => { event.preventDefault(); document.body.classList.add('dragging') })
for (const name of ['dragleave', 'drop']) window.addEventListener(name, event => { event.preventDefault(); document.body.classList.remove('dragging') })
window.addEventListener('drop', event => loadRom(event.dataTransfer.files[0]))
try {
  const response = await fetch('/api/status')
  if (!response.ok) throw new Error('Agent status unavailable.')
  const status = await response.json()
  available = status.available
  $('#agentDot').classList.toggle('on', available)
  $('#agentStatus').textContent = status.message
} catch { $('#agentStatus').textContent = 'Could not reach the local agent server. Reload to retry.' }
controls()
