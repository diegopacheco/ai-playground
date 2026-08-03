const canvas = document.getElementById('game')
const ctx = canvas.getContext('2d')
const scoreEl = document.getElementById('score')
const hitsEl = document.getElementById('hits')
const timeEl = document.getElementById('time')
const chainEl = document.getElementById('chain')
const gameCard = document.getElementById('game-card')
const cardTitle = document.getElementById('card-title')
const cardCopy = document.getElementById('card-copy')
const cardAction = document.getElementById('card-action')
const stopButton = document.getElementById('stop')
const calibrateButton = document.getElementById('calibrate')
const statusLight = document.getElementById('status-light')
const controllerStatus = document.getElementById('controller-status')
const hitFlash = document.getElementById('hit-flash')
const udpPortEl = document.getElementById('udp-port')
const pairingQr = document.getElementById('pairing-qr')
const pairingAddress = document.getElementById('pairing-address')
const pairingState = document.getElementById('pairing-state')
const staticMode = document.querySelector('meta[name="fly-catcher-mode"]')?.content === 'static'

const width = canvas.width
const height = canvas.height
const state = {
  mode: 'idle',
  score: 0,
  hits: 0,
  chain: 1,
  time: 60,
  startedAt: 0,
  lastFrame: performance.now(),
  lastPacketAt: 0,
  lastSnapAt: 0,
  motion: { ax: 0, ay: 0, az: 0 },
  center: { ax: 0, ay: 0 },
  aim: { x: width / 2, y: height / 2 },
  targetAim: { x: width / 2, y: height / 2 },
  fly: { x: 650, y: 190, vx: 130, vy: 75, phase: 0 },
  particles: [],
  swat: 0,
  miss: 0
}

function rect(x, y, w, h, color) {
  ctx.fillStyle = color
  ctx.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h))
}

function line(x1, y1, x2, y2, size, color) {
  ctx.strokeStyle = color
  ctx.lineWidth = size
  ctx.beginPath()
  ctx.moveTo(Math.round(x1), Math.round(y1))
  ctx.lineTo(Math.round(x2), Math.round(y2))
  ctx.stroke()
}

function drawKitchen() {
  rect(0, 0, width, height, '#91d6c8')
  for (let y = 0; y < 310; y += 42) {
    for (let x = (y / 42) % 2 ? -42 : 0; x < width; x += 84) {
      rect(x, y, 82, 40, '#a7e3d1')
      rect(x, y + 38, 82, 3, '#69b5ad')
      rect(x + 80, y, 3, 40, '#69b5ad')
    }
  }

  rect(52, 35, 250, 150, '#233c58')
  rect(62, 45, 230, 130, '#73bce8')
  rect(172, 45, 10, 130, '#fff0c9')
  rect(62, 104, 230, 9, '#fff0c9')
  rect(75, 58, 84, 38, '#c4eef1')
  rect(195, 58, 84, 38, '#c4eef1')
  rect(75, 124, 84, 38, '#9fd7dc')
  rect(195, 124, 84, 38, '#9fd7dc')
  rect(42, 185, 270, 9, '#5f493d')
  rect(42, 194, 270, 11, '#fff0c9')

  rect(696, 20, 224, 52, '#f8e0a6')
  rect(704, 29, 98, 35, '#ffb859')
  rect(810, 29, 101, 35, '#ffb859')
  rect(752, 44, 8, 8, '#664051')
  rect(854, 44, 8, 8, '#664051')
  rect(696, 76, 224, 14, '#63434b')

  rect(0, 305, width, 26, '#4d3645')
  rect(0, 331, width, 209, '#cc704f')
  for (let x = 0; x < width; x += 96) {
    rect(x + 3, 338, 90, 195, x % 192 === 0 ? '#d77c58' : '#c7674a')
    rect(x + 9, 351, 78, 174, '#b95c48')
    rect(x + 17, 359, 62, 158, x % 192 === 0 ? '#db865d' : '#cc704f')
    rect(x + 67, 425, 8, 8, '#4d3645')
  }

  rect(0, 293, width, 20, '#f6d58a')
  rect(0, 313, width, 16, '#8a4f45')
  rect(348, 259, 280, 39, '#e9c77e')
  rect(365, 268, 246, 23, '#6db2bc')
  rect(450, 231, 76, 30, '#8a4f45')
  rect(478, 208, 20, 52, '#f6d58a')
  rect(498, 208, 52, 12, '#f6d58a')
  rect(540, 213, 11, 32, '#f6d58a')

  rect(36, 235, 105, 58, '#f6d58a')
  rect(44, 225, 89, 17, '#fff0c9')
  rect(57, 191, 13, 34, '#384f44')
  rect(77, 200, 13, 25, '#384f44')
  rect(99, 184, 13, 41, '#384f44')
  rect(49, 192, 28, 10, '#65d6a7')
  rect(70, 192, 27, 10, '#65d6a7')
  rect(90, 180, 31, 10, '#65d6a7')

  rect(704, 211, 210, 82, '#3e4652')
  rect(718, 226, 182, 50, '#242b35')
  for (let x = 735; x <= 855; x += 40) {
    ctx.strokeStyle = '#687989'
    ctx.lineWidth = 6
    ctx.beginPath()
    ctx.arc(x, 251, 14, 0, Math.PI * 2)
    ctx.stroke()
  }
  rect(732, 281, 10, 10, '#ff5b4d')
  rect(760, 281, 10, 10, '#ffd447')

  rect(822, 111, 76, 84, '#fff0c9')
  rect(831, 102, 58, 17, '#ff5b4d')
  rect(839, 126, 42, 55, '#e8b957')
  rect(847, 138, 26, 8, '#5d453f')

  rect(0, 524, width, 16, '#6f3f3d')
  rect(158, 374, 9, 125, '#3e2a32')
  rect(147, 491, 31, 12, '#3e2a32')
  rect(793, 362, 9, 138, '#3e2a32')
  rect(782, 491, 31, 12, '#3e2a32')

  rect(286, 446, 25, 78, '#e2c697')
  rect(649, 438, 25, 86, '#e2c697')
  rect(278, 438, 41, 13, '#5d453f')
  rect(641, 430, 41, 13, '#5d453f')
}

function drawFly() {
  const fly = state.fly
  const bob = Math.sin(fly.phase * 4) * 4
  const x = Math.round(fly.x)
  const y = Math.round(fly.y + bob)
  const wingUp = Math.sin(fly.phase * 18) > 0
  rect(x - 5, y - 2, 18, 14, '#141019')
  rect(x + 12, y + 1, 10, 8, '#2b202f')
  rect(x - 9, y + 1, 5, 5, '#ff5b4d')
  rect(x + 4, y + 1, 5, 5, '#d62f3f')
  rect(x - 14, y + (wingUp ? -14 : -7), 14, 10, '#e8fff1')
  rect(x + 8, y + (wingUp ? -14 : -7), 14, 10, '#e8fff1')
  rect(x - 12, y + (wingUp ? -11 : -4), 10, 6, '#86c7d5')
  rect(x + 10, y + (wingUp ? -11 : -4), 10, 6, '#86c7d5')
  line(x - 3, y + 12, x - 11, y + 19, 3, '#141019')
  line(x + 8, y + 12, x + 16, y + 19, 3, '#141019')
}

function drawAim() {
  const x = Math.round(state.aim.x)
  const y = Math.round(state.aim.y)
  const squeeze = state.swat > 0 ? 10 : 0
  ctx.save()
  ctx.translate(x, y)
  ctx.rotate(-0.16)
  rect(-3, 29 - squeeze, 7, 170, '#3b2630')
  rect(-7, 28 - squeeze, 15, 17, '#ffd447')
  ctx.strokeStyle = state.miss > 0 ? '#ff5b4d' : '#fff0c9'
  ctx.lineWidth = 7
  ctx.strokeRect(-42 + squeeze / 2, -39 + squeeze / 2, 84 - squeeze, 76 - squeeze)
  ctx.strokeStyle = '#563747'
  ctx.lineWidth = 3
  for (let offset = -28; offset <= 28; offset += 14) {
    line(offset, -35 + squeeze / 2, offset, 33 - squeeze / 2, 3, '#563747')
    line(-38 + squeeze / 2, offset, 38 - squeeze / 2, offset, 3, '#563747')
  }
  ctx.restore()
}

function drawParticles() {
  for (const particle of state.particles) {
    rect(particle.x, particle.y, particle.size, particle.size, particle.color)
  }
}

function moveFly(delta) {
  const fly = state.fly
  fly.phase += delta
  fly.x += fly.vx * delta
  fly.y += fly.vy * delta
  fly.vx += Math.sin(fly.phase * 2.3) * delta * 85
  fly.vy += Math.cos(fly.phase * 3.1) * delta * 70
  const speed = Math.hypot(fly.vx, fly.vy)
  const maxSpeed = 205 + state.hits * 3
  if (speed > maxSpeed) {
    fly.vx = fly.vx / speed * maxSpeed
    fly.vy = fly.vy / speed * maxSpeed
  }
  if (fly.x < 28 || fly.x > width - 40) fly.vx *= -1
  if (fly.y < 100 || fly.y > height - 120) fly.vy *= -1
  fly.x = Math.max(28, Math.min(width - 40, fly.x))
  fly.y = Math.max(100, Math.min(height - 120, fly.y))
}

function updateAim(delta) {
  const speed = Math.min(1, delta * 14)
  state.aim.x += (state.targetAim.x - state.aim.x) * speed
  state.aim.y += (state.targetAim.y - state.aim.y) * speed
}

function updateParticles(delta) {
  for (const particle of state.particles) {
    particle.x += particle.vx * delta
    particle.y += particle.vy * delta
    particle.vy += 180 * delta
    particle.life -= delta
  }
  state.particles = state.particles.filter((particle) => particle.life > 0)
}

function updateScoreboard() {
  scoreEl.textContent = String(state.score).padStart(6, '0')
  hitsEl.textContent = String(state.hits).padStart(2, '0')
  timeEl.textContent = String(Math.max(0, Math.ceil(state.time))).padStart(2, '0')
  chainEl.textContent = `x${state.chain}`
}

function makeBurst(x, y, hit) {
  const colors = hit ? ['#ffd447', '#fff0c9', '#ff5b4d'] : ['#ff5b4d', '#563747']
  for (let index = 0; index < 18; index += 1) {
    const angle = Math.PI * 2 * index / 18
    const force = 70 + Math.random() * 130
    state.particles.push({
      x,
      y,
      vx: Math.cos(angle) * force,
      vy: Math.sin(angle) * force,
      life: 0.35 + Math.random() * 0.35,
      size: 4 + Math.floor(Math.random() * 6),
      color: colors[index % colors.length]
    })
  }
}

function placeFly() {
  const angle = Math.random() * Math.PI * 2
  const speed = 135 + state.hits * 4
  state.fly.x = 100 + Math.random() * (width - 200)
  state.fly.y = 110 + Math.random() * (height - 250)
  state.fly.vx = Math.cos(angle) * speed
  state.fly.vy = Math.sin(angle) * speed
}

function playTone(frequency, duration) {
  const AudioContext = window.AudioContext || window.webkitAudioContext
  if (!AudioContext) return
  const audio = playTone.audio || new AudioContext()
  playTone.audio = audio
  const oscillator = audio.createOscillator()
  const gain = audio.createGain()
  oscillator.type = 'square'
  oscillator.frequency.value = frequency
  gain.gain.setValueAtTime(0.06, audio.currentTime)
  gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + duration)
  oscillator.connect(gain)
  gain.connect(audio.destination)
  oscillator.start()
  oscillator.stop(audio.currentTime + duration)
}

function snap() {
  const now = performance.now()
  if (state.mode !== 'running' || now - state.lastSnapAt < 220) return
  state.lastSnapAt = now
  state.swat = 0.16
  const distance = Math.hypot(state.aim.x - state.fly.x, state.aim.y - state.fly.y)
  if (distance <= 62) {
    const gained = 100 + (state.chain - 1) * 25
    state.score += gained
    state.hits += 1
    state.chain = Math.min(9, state.chain + 1)
    makeBurst(state.fly.x, state.fly.y, true)
    placeFly()
    hitFlash.classList.remove('active')
    void hitFlash.offsetWidth
    hitFlash.classList.add('active')
    playTone(620, 0.09)
    window.setTimeout(() => playTone(880, 0.08), 70)
  } else {
    state.chain = 1
    state.score = Math.max(0, state.score - 25)
    state.miss = 0.25
    makeBurst(state.aim.x, state.aim.y, false)
    playTone(105, 0.12)
  }
  updateScoreboard()
}

function calibrate() {
  state.center.ax = state.motion.ax
  state.center.ay = state.motion.ay
  state.targetAim.x = width / 2
  state.targetAim.y = height / 2
}

function startRound() {
  if (!staticMode && Date.now() - state.lastPacketAt >= 2200) return
  state.mode = 'running'
  state.score = 0
  state.hits = 0
  state.chain = 1
  state.time = 60
  state.startedAt = performance.now()
  state.lastFrame = performance.now()
  state.particles = []
  calibrate()
  placeFly()
  gameCard.classList.add('hidden')
  stopButton.disabled = false
  updateScoreboard()
  playTone(330, 0.08)
}

function endRound(stopped) {
  if (state.mode !== 'running') return
  state.mode = 'ended'
  stopButton.disabled = true
  cardTitle.textContent = stopped ? 'Round stopped' : 'Kitchen clear'
  cardCopy.textContent = `Final score ${state.score}. You caught ${state.hits} ${state.hits === 1 ? 'fly' : 'flies'}.`
  cardAction.textContent = 'Play again'
  cardAction.disabled = !staticMode && Date.now() - state.lastPacketAt >= 2200
  gameCard.classList.remove('hidden')
  playTone(stopped ? 180 : 740, 0.15)
}

function handleMotion(packet) {
  state.motion.ax = packet.ax
  state.motion.ay = packet.ay
  state.motion.az = packet.az
  state.lastPacketAt = Date.now()
  const dx = packet.ax - state.center.ax
  const dy = packet.ay - state.center.ay
  state.targetAim.x = Math.max(35, Math.min(width - 35, width / 2 + dx * 610))
  state.targetAim.y = Math.max(45, Math.min(height - 80, height / 2 - dy * 520))
}

function confirmController() {
  state.lastPacketAt = Date.now()
  pairingState.textContent = 'Controller linked by UDP'
  pairingState.classList.add('online')
  if (state.mode !== 'running') cardAction.disabled = false
  if (state.mode === 'idle') {
    cardTitle.textContent = 'Controller linked'
    cardCopy.textContent = 'Connection confirmed. Keep the iPhone centered, then start the round.'
  }
}

function showMissingAddress() {
  const qrContext = pairingQr.getContext('2d')
  pairingQr.width = 225
  pairingQr.height = 225
  qrContext.fillStyle = '#fff0c9'
  qrContext.fillRect(0, 0, 225, 225)
  qrContext.fillStyle = '#17131e'
  qrContext.font = '900 24px monospace'
  qrContext.textAlign = 'center'
  qrContext.fillText('NO LAN', 112, 120)
  pairingAddress.textContent = 'Private IPv4 unavailable'
  cardCopy.textContent = 'Connect the Mac and iPhone to the same private Wi-Fi network, then restart.'
}

function draw() {
  drawKitchen()
  drawParticles()
  if (state.mode === 'running') drawFly()
  drawAim()
}

function frame(now) {
  const delta = Math.min(0.034, (now - state.lastFrame) / 1000)
  state.lastFrame = now
  if (state.mode === 'running') {
    state.time = 60 - (now - state.startedAt) / 1000
    if (state.time <= 0) {
      state.time = 0
      endRound(false)
    } else {
      moveFly(delta)
    }
  }
  updateAim(delta)
  updateParticles(delta)
  state.swat = Math.max(0, state.swat - delta)
  state.miss = Math.max(0, state.miss - delta)
  updateScoreboard()
  draw()
  requestAnimationFrame(frame)
}

if (staticMode) {
  state.lastPacketAt = Date.now()
  statusLight.classList.add('online')
  controllerStatus.textContent = 'Keyboard ready'
  pairingState.textContent = 'Arrow keys and Space ready'
  pairingState.classList.add('online')
  cardAction.disabled = false
} else {
  const events = new EventSource('/events')
  events.onmessage = (event) => {
    let packet
    try {
      packet = JSON.parse(event.data)
    } catch {
      return
    }
    confirmController()
    if (packet.type === 'motion') handleMotion(packet)
    if (packet.type === 'snap') snap()
  }

  window.setInterval(() => {
    const online = Date.now() - state.lastPacketAt < 2200
    statusLight.classList.toggle('online', online)
    controllerStatus.textContent = online ? 'iPhone controller linked' : 'Waiting for controller'
    if (!online) {
      pairingState.textContent = 'Waiting for UDP signal'
      pairingState.classList.remove('online')
      if (state.mode !== 'running') cardAction.disabled = true
      if (state.mode === 'idle') {
        cardTitle.textContent = 'Scan to link'
        cardCopy.textContent = 'Install Controller 1.1 from Xcode, scan with Camera, then open the installed controller.'
      }
      if (state.mode === 'ended') {
        cardTitle.textContent = 'Controller offline'
        cardCopy.textContent = 'Scan the code and wait for the UDP link before starting another round.'
      }
    }
  }, 500)

  fetch('/api/status')
    .then((response) => response.json())
    .then((status) => {
      const port = status.udp.split(':').pop()
      udpPortEl.textContent = `UDP ${port}`
      if (!status.pairingUrl) {
        showMissingAddress()
        return
      }
      window.renderPairingQr(pairingQr, status.pairingUrl)
      pairingAddress.textContent = `${status.controllerHost}:${port}`
    })
    .catch(showMissingAddress)
}
requestAnimationFrame(frame)
