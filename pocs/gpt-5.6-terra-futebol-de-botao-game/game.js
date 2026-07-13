const canvas = document.getElementById('game')
const context = canvas.getContext('2d')
const blueScoreElement = document.getElementById('blueScore')
const redScoreElement = document.getElementById('redScore')
const clockElement = document.getElementById('clock')
const statusText = document.getElementById('statusText')
const statusDot = document.getElementById('statusDot')
const turnDisc = document.getElementById('turnDisc')
const turnName = document.getElementById('turnName')
const turnHint = document.getElementById('turnHint')
const startButton = document.getElementById('startButton')
const resetButton = document.getElementById('resetButton')
const soundButton = document.getElementById('soundButton')
const fullscreenButton = document.getElementById('fullscreenButton')
const goalBanner = document.getElementById('goalBanner')
const setupModal = document.getElementById('setupModal')
const setupTitle = document.getElementById('setupTitle')
const setupCopy = document.getElementById('setupCopy')
const modeOptions = document.getElementById('modeOptions')
const agentOptions = document.getElementById('agentOptions')
const narrationOptions = document.getElementById('narrationOptions')
const narratorAgentOptions = document.getElementById('narratorAgentOptions')
const languageOptions = document.getElementById('languageOptions')
const backButton = document.getElementById('backButton')
const commentaryBubble = document.getElementById('commentaryBubble')

const width = canvas.width
const height = canvas.height
const goalTop = 247
const goalBottom = 453
const playerRadius = 34
const ballRadius = 15
const friction = 0.985
const stopSpeed = 0.06
const maxPower = 30

let players = []
let ball = {}
let selected = null
let pointer = null
let turn = 'blue'
let blueScore = 0
let redScore = 0
let timeLeft = 180
let running = false
let paused = false
let moving = false
let soundOn = true
let lastTime = 0
let clockAccumulator = 0
let audioContext = null
let crowdSource = null
let crowdGain = null
let crowdLfo = null
let gameMode = null
let aiProvider = null
let aiThinking = false
let aiRequestId = 0
let confetti = []
let setupStage = 'mode'
let pendingGameMode = null
let pendingGameProvider = null
let narrationOn = false
let narratorProvider = null
let narrationLanguage = null
let lastShot = null
let commentaryQueue = Promise.resolve()

const colors = {
  blue: '#1975c9',
  red: '#df3f2e'
}

function createPlayer(x, y, team, number, name) {
  return { x, y, vx: 0, vy: 0, team, number, name, radius: playerRadius }
}

function resetPositions() {
  players = [
    createPlayer(175, 350, 'blue', 1, 'Danrlei'),
    createPlayer(320, 175, 'blue', 2, 'Arce'),
    createPlayer(320, 525, 'blue', 3, 'Dinho'),
    createPlayer(465, 280, 'blue', 4, 'C. Miguel'),
    createPlayer(465, 420, 'blue', 5, 'P. Nunes'),
    createPlayer(1025, 350, 'red', 1, 'André'),
    createPlayer(880, 175, 'red', 2, 'Gamarra'),
    createPlayer(880, 525, 'red', 3, 'Enciso'),
    createPlayer(735, 280, 'red', 4, 'Arílson'),
    createPlayer(735, 420, 'red', 5, 'Fabiano')
  ]
  ball = { x: 600, y: 350, vx: 0, vy: 0, radius: ballRadius }
  moving = false
  selected = null
  pointer = null
}

function resetGame() {
  aiRequestId += 1
  aiThinking = false
  confetti = []
  lastShot = null
  blueScore = 0
  redScore = 0
  timeLeft = 180
  turn = 'blue'
  running = false
  paused = false
  startButton.textContent = 'INICIAR JOGO'
  statusText.textContent = 'Pronto para jogar'
  statusDot.classList.remove('live')
  resetPositions()
  updateInterface()
}

function toggleGame() {
  if (timeLeft <= 0) resetGame()
  running = true
  paused = !paused
  if (startButton.textContent === 'INICIAR JOGO') paused = false
  startButton.textContent = paused ? 'CONTINUAR' : 'PAUSAR'
  statusText.textContent = paused ? 'Partida pausada' : 'Partida em andamento'
  statusDot.classList.toggle('live', !paused)
  lastTime = performance.now()
  beep(440, .05)
  startCrowd()
  requestAiTurn()
}

function updateInterface() {
  blueScoreElement.textContent = blueScore
  redScoreElement.textContent = redScore
  const minutes = Math.floor(timeLeft / 60)
  const seconds = Math.max(0, Math.ceil(timeLeft % 60))
  clockElement.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
  turnDisc.className = `turn-disc ${turn}`
  turnName.textContent = turn === 'blue' ? 'Grêmio' : 'Inter'
  if (!running) turnHint.textContent = 'Inicie a partida para jogar.'
  else if (paused) turnHint.textContent = 'A partida está pausada.'
  else if (aiThinking) turnHint.textContent = `${providerName()} está escolhendo a jogada.`
  else if (moving) turnHint.textContent = 'Aguarde os botões pararem.'
  else turnHint.textContent = 'Puxe um botão para trás e solte.'
}

function beep(frequency, duration, force = false, delay = 0) {
  if (!soundOn && !force) return
  const AudioEngine = window.AudioContext || window.webkitAudioContext
  if (!AudioEngine) return
  audioContext ||= new AudioEngine()
  if (audioContext.state === 'suspended') audioContext.resume()
  const oscillator = audioContext.createOscillator()
  const gain = audioContext.createGain()
  const start = audioContext.currentTime + delay
  oscillator.frequency.value = frequency
  oscillator.type = 'triangle'
  gain.gain.setValueAtTime(.18, start)
  gain.gain.exponentialRampToValueAtTime(.001, start + duration)
  oscillator.connect(gain)
  gain.connect(audioContext.destination)
  oscillator.start(start)
  oscillator.stop(start + duration)
}

function startCrowd() {
  if (!soundOn || crowdSource) return
  const AudioEngine = window.AudioContext || window.webkitAudioContext
  if (!AudioEngine) return
  audioContext ||= new AudioEngine()
  if (audioContext.state === 'suspended') audioContext.resume()
  const buffer = audioContext.createBuffer(1, audioContext.sampleRate * 3, audioContext.sampleRate)
  const data = buffer.getChannelData(0)
  let level = 0
  for (let index = 0; index < data.length; index += 1) {
    level = level * .94 + (Math.random() * 2 - 1) * .06
    data[index] = level
  }
  crowdSource = audioContext.createBufferSource()
  crowdGain = audioContext.createGain()
  crowdLfo = audioContext.createOscillator()
  const lfoDepth = audioContext.createGain()
  const filter = audioContext.createBiquadFilter()
  crowdSource.buffer = buffer
  crowdSource.loop = true
  filter.type = 'bandpass'
  filter.frequency.value = 620
  filter.Q.value = .35
  crowdGain.gain.value = .08
  crowdLfo.frequency.value = .11
  lfoDepth.gain.value = .018
  crowdLfo.connect(lfoDepth)
  lfoDepth.connect(crowdGain.gain)
  crowdSource.connect(filter)
  filter.connect(crowdGain)
  crowdGain.connect(audioContext.destination)
  crowdSource.start()
  crowdLfo.start()
}

function stopCrowd() {
  if (!crowdSource) return
  crowdSource.stop()
  crowdLfo.stop()
  crowdSource = null
  crowdGain = null
  crowdLfo = null
}

function cheer() {
  if (!soundOn || !crowdGain) return
  const now = audioContext.currentTime
  crowdGain.gain.cancelScheduledValues(now)
  crowdGain.gain.setValueAtTime(.08, now)
  crowdGain.gain.linearRampToValueAtTime(.25, now + .15)
  crowdGain.gain.exponentialRampToValueAtTime(.08, now + 1.8)
}

function providerName() {
  return { claude: 'Claude', codex: 'Codex', agy: 'Agy' }[aiProvider] || 'IA'
}

function hideSetupOptions() {
  modeOptions.hidden = true
  agentOptions.hidden = true
  narrationOptions.hidden = true
  narratorAgentOptions.hidden = true
  languageOptions.hidden = true
}

function closeSetup() {
  gameMode = pendingGameMode
  aiProvider = pendingGameProvider
  setupModal.classList.add('closed')
  resetGame()
  startCrowd()
}

function showAgentOptions() {
  setupStage = 'game-agent'
  setupTitle.textContent = 'Escolha a IA'
  setupCopy.textContent = 'Quem vai comandar o Inter de 96?'
  hideSetupOptions()
  agentOptions.hidden = false
  backButton.hidden = false
}

function showModeOptions() {
  setupStage = 'mode'
  setupTitle.textContent = 'Quem vai jogar?'
  setupCopy.textContent = 'Escolha o formato da partida.'
  hideSetupOptions()
  modeOptions.hidden = false
  backButton.hidden = true
}

function showNarrationOptions() {
  setupStage = 'narration'
  setupTitle.textContent = 'Quer narração?'
  setupCopy.textContent = 'Um comentarista pode dar voz a cada jogada.'
  hideSetupOptions()
  narrationOptions.hidden = false
  backButton.hidden = false
}

function showNarratorAgents() {
  setupStage = 'narrator-agent'
  setupTitle.textContent = 'Escolha o narrador'
  setupCopy.textContent = 'Qual IA vai assumir o microfone?'
  hideSetupOptions()
  narratorAgentOptions.hidden = false
  backButton.hidden = false
}

function showLanguageOptions() {
  setupStage = 'language'
  setupTitle.textContent = 'Qual idioma?'
  setupCopy.textContent = 'Escolha a voz da transmissão.'
  hideSetupOptions()
  languageOptions.hidden = false
  backButton.hidden = false
}

function chooseGame(mode, provider = null) {
  pendingGameMode = mode
  pendingGameProvider = provider
  showNarrationOptions()
}

function fallbackAiShot() {
  const redPlayers = players.filter(player => player.team === 'red')
  const player = redPlayers.reduce((closest, current) => Math.hypot(current.x - ball.x, current.y - ball.y) < Math.hypot(closest.x - ball.x, closest.y - ball.y) ? current : closest)
  return { player: player.number, angle: Math.atan2(ball.y - player.y, ball.x - player.x), power: .72 }
}

async function requestAiTurn() {
  if (gameMode !== 'ai' || turn !== 'red' || !running || paused || moving || aiThinking) return
  aiThinking = true
  const requestId = ++aiRequestId
  statusText.textContent = `${providerName()} pensa pelo Inter`
  updateInterface()
  let shot
  try {
    const response = await fetch('/api/ai-shot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        provider: aiProvider,
        players: players.filter(player => player.team === 'red').map(({ number, x, y }) => ({ number, x, y })),
        ball: { x: ball.x, y: ball.y }
      })
    })
    if (!response.ok) throw new Error('AI unavailable')
    shot = await response.json()
  } catch {
    shot = fallbackAiShot()
    statusText.textContent = 'IA local indisponível · jogada automática'
  }
  if (requestId !== aiRequestId || !running || paused || turn !== 'red') {
    aiThinking = false
    updateInterface()
    return
  }
  const player = players.find(item => item.team === 'red' && item.number === Number(shot.player)) || players.find(item => item.team === 'red')
  const power = Math.max(.25, Math.min(1, Number(shot.power)))
  const angle = Number(shot.angle)
  player.vx = Math.cos(angle) * maxPower * power
  player.vy = Math.sin(angle) * maxPower * power
  aiThinking = false
  moving = true
  statusText.textContent = 'Partida em andamento'
  beep(180, .06)
  updateInterface()
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect()
  return {
    x: (event.clientX - rect.left) * width / rect.width,
    y: (event.clientY - rect.top) * height / rect.height
  }
}

function handlePointerDown(event) {
  if (!running || paused || moving || aiThinking || gameMode === 'ai' && turn === 'red') return
  const point = pointerPosition(event)
  const target = players.find(player => player.team === turn && Math.hypot(point.x - player.x, point.y - player.y) <= player.radius + 10)
  if (!target) return
  selected = target
  pointer = point
  canvas.setPointerCapture(event.pointerId)
}

function handlePointerMove(event) {
  if (!selected) return
  pointer = pointerPosition(event)
}

function handlePointerUp(event) {
  if (!selected || !pointer) return
  const dx = selected.x - pointer.x
  const dy = selected.y - pointer.y
  const distance = Math.hypot(dx, dy)
  if (distance > 12) {
    const power = Math.min(distance * .13, maxPower)
    selected.vx = dx / distance * power
    selected.vy = dy / distance * power
    moving = true
    beep(180, .06)
  }
  selected = null
  pointer = null
  canvas.releasePointerCapture(event.pointerId)
  updateInterface()
}

function collide(a, b) {
  const dx = b.x - a.x
  const dy = b.y - a.y
  const distance = Math.hypot(dx, dy)
  const minimum = a.radius + b.radius
  if (!distance || distance >= minimum) return
  const nx = dx / distance
  const ny = dy / distance
  const overlap = minimum - distance
  a.x -= nx * overlap / 2
  a.y -= ny * overlap / 2
  b.x += nx * overlap / 2
  b.y += ny * overlap / 2
  const relative = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
  if (relative >= 0) return
  const impulse = relative * .96
  a.vx += impulse * nx
  a.vy += impulse * ny
  b.vx -= impulse * nx
  b.vy -= impulse * ny
  if (Math.abs(relative) > 1.4) beep(110 + Math.min(Math.abs(relative) * 15, 160), .035)
}

function constrain(body, isBall) {
  const inGoal = body.y > goalTop && body.y < goalBottom
  if (!inGoal || !isBall) {
    if (body.x - body.radius < 45) {
      body.x = 45 + body.radius
      body.vx = Math.abs(body.vx) * .8
    }
    if (body.x + body.radius > width - 45) {
      body.x = width - 45 - body.radius
      body.vx = -Math.abs(body.vx) * .8
    }
  }
  if (body.y - body.radius < 45) {
    body.y = 45 + body.radius
    body.vy = Math.abs(body.vy) * .8
  }
  if (body.y + body.radius > height - 45) {
    body.y = height - 45 - body.radius
    body.vy = -Math.abs(body.vy) * .8
  }
}

function scoreGoal(team) {
  if (team === 'blue') blueScore += 1
  else redScore += 1
  goalBanner.classList.remove('show')
  void goalBanner.offsetWidth
  goalBanner.classList.add('show')
  beep(660, .3)
  setTimeout(() => beep(880, .35), 180)
  cheer()
  spawnConfetti(team)
  turn = team === 'blue' ? 'red' : 'blue'
  resetPositions()
  updateInterface()
  setTimeout(requestAiTurn, 700)
}

function spawnConfetti(team) {
  const palette = team === 'blue' ? ['#1975c9', '#111111', '#f3ead4'] : ['#df3f2e', '#f3ead4', '#f4c644']
  confetti = Array.from({ length: 150 }, (_, index) => ({
    x: width / 2 + (Math.random() - .5) * 180,
    y: height / 2,
    vx: (Math.random() - .5) * 20,
    vy: -Math.random() * 20 - 5,
    gravity: .25 + Math.random() * .2,
    rotation: Math.random() * Math.PI,
    spin: (Math.random() - .5) * .3,
    size: 5 + Math.random() * 9,
    color: palette[index % palette.length],
    life: 130 + Math.random() * 50
  }))
}

function drawConfetti() {
  confetti = confetti.filter(piece => piece.life > 0)
  for (const piece of confetti) {
    piece.x += piece.vx
    piece.y += piece.vy
    piece.vy += piece.gravity
    piece.vx *= .99
    piece.rotation += piece.spin
    piece.life -= 1
    context.save()
    context.translate(piece.x, piece.y)
    context.rotate(piece.rotation)
    context.fillStyle = piece.color
    context.globalAlpha = Math.min(1, piece.life / 30)
    context.fillRect(-piece.size / 2, -piece.size / 4, piece.size, piece.size / 2)
    context.restore()
  }
}

function updatePhysics() {
  const bodies = [...players, ball]
  for (const body of bodies) {
    body.x += body.vx
    body.y += body.vy
    body.vx *= friction
    body.vy *= friction
    if (Math.hypot(body.vx, body.vy) < stopSpeed) {
      body.vx = 0
      body.vy = 0
    }
    constrain(body, body === ball)
  }
  for (let index = 0; index < bodies.length; index += 1) {
    for (let other = index + 1; other < bodies.length; other += 1) collide(bodies[index], bodies[other])
  }
  if (ball.x < 18 && ball.y > goalTop && ball.y < goalBottom) scoreGoal('red')
  if (ball.x > width - 18 && ball.y > goalTop && ball.y < goalBottom) scoreGoal('blue')
  if (moving && bodies.every(body => body.vx === 0 && body.vy === 0)) {
    moving = false
    turn = turn === 'blue' ? 'red' : 'blue'
    updateInterface()
    requestAiTurn()
  }
}

function drawField() {
  const gradient = context.createLinearGradient(0, 0, 0, height)
  gradient.addColorStop(0, '#126944')
  gradient.addColorStop(1, '#0b4b31')
  context.fillStyle = gradient
  context.fillRect(0, 0, width, height)
  for (let x = 0; x < width; x += 120) {
    context.fillStyle = x % 240 === 0 ? 'rgba(255,255,255,.018)' : 'rgba(0,0,0,.018)'
    context.fillRect(x, 0, 120, height)
  }
  context.strokeStyle = 'rgba(245,238,211,.9)'
  context.lineWidth = 5
  context.strokeRect(45, 45, width - 90, height - 90)
  context.beginPath()
  context.moveTo(width / 2, 45)
  context.lineTo(width / 2, height - 45)
  context.stroke()
  context.beginPath()
  context.arc(width / 2, height / 2, 105, 0, Math.PI * 2)
  context.stroke()
  context.fillStyle = 'rgba(245,238,211,.9)'
  context.beginPath()
  context.arc(width / 2, height / 2, 6, 0, Math.PI * 2)
  context.fill()
  context.strokeRect(45, 185, 165, 330)
  context.strokeRect(width - 210, 185, 165, 330)
  context.strokeRect(12, goalTop, 33, goalBottom - goalTop)
  context.strokeRect(width - 45, goalTop, 33, goalBottom - goalTop)
}

function drawPlayer(player) {
  context.save()
  context.translate(player.x, player.y)
  context.shadowColor = 'rgba(0,0,0,.38)'
  context.shadowBlur = 8
  context.shadowOffsetY = 5
  context.fillStyle = colors[player.team]
  context.beginPath()
  context.arc(0, 0, player.radius, 0, Math.PI * 2)
  context.fill()
  context.shadowColor = 'transparent'
  context.strokeStyle = '#f7efd9'
  context.lineWidth = 6
  context.stroke()
  context.strokeStyle = 'rgba(0,0,0,.2)'
  context.lineWidth = 2
  context.beginPath()
  context.arc(0, 0, player.radius - 8, 0, Math.PI * 2)
  context.stroke()
  context.fillStyle = '#f7efd9'
  context.font = '800 13px Barlow Condensed'
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  context.fillText(player.name, 0, 3)
  context.font = '800 9px Barlow Condensed'
  context.fillText(player.number, 0, -12)
  context.restore()
}

function drawBall() {
  context.save()
  context.translate(ball.x, ball.y)
  context.shadowColor = 'rgba(0,0,0,.4)'
  context.shadowBlur = 5
  context.shadowOffsetY = 3
  context.fillStyle = '#f5d949'
  context.beginPath()
  context.arc(0, 0, ball.radius, 0, Math.PI * 2)
  context.fill()
  context.shadowColor = 'transparent'
  context.strokeStyle = '#3b3215'
  context.lineWidth = 2
  context.stroke()
  context.fillStyle = '#3b3215'
  context.beginPath()
  context.arc(0, 0, 4, 0, Math.PI * 2)
  context.fill()
  context.restore()
}

function drawAim() {
  if (!selected || !pointer) return
  const dx = selected.x - pointer.x
  const dy = selected.y - pointer.y
  const distance = Math.min(Math.hypot(dx, dy), 230)
  const angle = Math.atan2(dy, dx)
  context.save()
  context.setLineDash([10, 8])
  context.strokeStyle = '#f4c644'
  context.lineWidth = 5
  context.beginPath()
  context.moveTo(selected.x, selected.y)
  context.lineTo(selected.x + Math.cos(angle) * distance, selected.y + Math.sin(angle) * distance)
  context.stroke()
  context.setLineDash([])
  context.fillStyle = 'rgba(244,198,68,.22)'
  context.beginPath()
  context.arc(selected.x, selected.y, selected.radius + distance * .1, 0, Math.PI * 2)
  context.fill()
  context.restore()
}

function draw() {
  drawField()
  players.forEach(drawPlayer)
  drawBall()
  drawAim()
  drawConfetti()
}

function loop(timestamp) {
  const delta = Math.min(timestamp - lastTime, 50)
  lastTime = timestamp
  if (running && !paused) {
    updatePhysics()
    if (!aiThinking) clockAccumulator += delta
    if (!aiThinking && clockAccumulator >= 1000) {
      timeLeft = Math.max(0, timeLeft - clockAccumulator / 1000)
      clockAccumulator = 0
      updateInterface()
      if (timeLeft <= 0) {
        paused = true
        moving = false
        statusText.textContent = 'Fim de jogo'
        statusDot.classList.remove('live')
        startButton.textContent = 'NOVO JOGO'
        turnHint.textContent = blueScore === redScore ? 'Empate na mesa.' : `${blueScore > redScore ? 'Grêmio' : 'Inter'} venceu!`
        beep(220, .5)
      }
    }
  }
  draw()
  requestAnimationFrame(loop)
}

startButton.addEventListener('click', toggleGame)
resetButton.addEventListener('click', resetGame)
soundButton.addEventListener('click', () => {
  if (soundOn) {
    beep(520, .08, true)
    beep(330, .12, true, .09)
    soundOn = false
    stopCrowd()
  } else {
    soundOn = true
    beep(440, .08)
    beep(660, .14, false, .09)
    startCrowd()
  }
  soundButton.textContent = `SOM ${soundOn ? 'LIGADO' : 'DESLIGADO'}`
  soundButton.setAttribute('aria-label', `${soundOn ? 'Desativar' : 'Ativar'} som`)
})
document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
  if (button.dataset.mode === 'ai') showAgentOptions()
  else closeSetup('human')
}))
document.querySelectorAll('[data-provider]').forEach(button => button.addEventListener('click', () => closeSetup('ai', button.dataset.provider)))
backButton.addEventListener('click', showModeOptions)
fullscreenButton.addEventListener('click', () => {
  const enter = document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen
  const exit = document.exitFullscreen || document.webkitExitFullscreen
  const active = document.fullscreenElement || document.webkitFullscreenElement
  if (active) exit.call(document)
  else if (enter) enter.call(document.documentElement)
})
function updateFullscreenButton() {
  const active = Boolean(document.fullscreenElement || document.webkitFullscreenElement)
  fullscreenButton.textContent = active ? 'SAIR DA TELA' : 'TELA CHEIA'
  fullscreenButton.setAttribute('aria-label', active ? 'Sair da tela cheia' : 'Entrar em tela cheia')
}
document.addEventListener('fullscreenchange', updateFullscreenButton)
document.addEventListener('webkitfullscreenchange', updateFullscreenButton)
canvas.addEventListener('pointerdown', handlePointerDown)
canvas.addEventListener('pointermove', handlePointerMove)
canvas.addEventListener('pointerup', handlePointerUp)
canvas.addEventListener('pointercancel', handlePointerUp)

resetGame()
requestAnimationFrame(loop)
