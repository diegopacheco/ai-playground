const token = new URLSearchParams(location.search).get('token') || ''
const enableButton = document.getElementById('enable')
const snapButton = document.getElementById('snap')
const statusLight = document.getElementById('status-light')
const statusText = document.getElementById('status-text')
const axisX = document.getElementById('axis-x')
const axisY = document.getElementById('axis-y')
const axisZ = document.getElementById('axis-z')
const tiltDot = document.getElementById('tilt-dot')
let active = false
let motionInFlight = false
let pendingMotion
let lastMotionAt = 0
let lastSnapAt = 0
let wakeLock

function setStatus(text, online) {
  statusText.textContent = text
  statusLight.classList.toggle('online', online)
}

async function send(packet) {
  try {
    const response = await fetch('/api/control', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Fly-Token': token
      },
      body: JSON.stringify(packet)
    })
    if (!response.ok) throw new Error('Link rejected')
    setStatus('Linked to the kitchen', true)
  } catch {
    setStatus('Local controller link lost', false)
  }
}

async function flushMotion() {
  if (motionInFlight || !pendingMotion) return
  motionInFlight = true
  const packet = pendingMotion
  pendingMotion = null
  await send(packet)
  motionInFlight = false
  flushMotion()
}

function queueMotion(packet) {
  pendingMotion = packet
  flushMotion()
}

function updateMotion(event) {
  const now = performance.now()
  if (!active || now - lastMotionAt < 16) return
  const acceleration = event.accelerationIncludingGravity || event.acceleration
  if (!acceleration || acceleration.x == null || acceleration.y == null || acceleration.z == null) return
  lastMotionAt = now
  const ax = acceleration.x / 9.80665
  const ay = acceleration.y / 9.80665
  const az = acceleration.z / 9.80665
  axisX.textContent = ax.toFixed(2)
  axisY.textContent = ay.toFixed(2)
  axisZ.textContent = az.toFixed(2)
  const x = Math.max(-84, Math.min(84, ax * 96))
  const y = Math.max(-84, Math.min(84, -ay * 96))
  tiltDot.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`
  queueMotion({ type: 'motion', ax, ay, az })
  const force = Math.hypot(ax, ay, az)
  if (force > 1.72 && now - lastSnapAt > 550) {
    lastSnapAt = now
    send({ type: 'snap' })
  }
}

async function enableMotion() {
  if (!window.isSecureContext) {
    setStatus('Trust the local certificate first', false)
    return
  }
  if (typeof DeviceMotionEvent === 'undefined') {
    setStatus('Motion sensor unavailable', false)
    return
  }
  try {
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
      const permission = await DeviceMotionEvent.requestPermission()
      if (permission !== 'granted') {
        setStatus('Motion permission denied', false)
        return
      }
    }
    active = true
    enableButton.textContent = 'Motion enabled'
    enableButton.disabled = true
    snapButton.disabled = false
    window.addEventListener('devicemotion', updateMotion)
    if ('wakeLock' in navigator) {
      try {
        wakeLock = await navigator.wakeLock.request('screen')
      } catch {
      }
    }
    await send({ type: 'motion', ax: 0, ay: 0, az: 1 })
  } catch {
    setStatus('Motion permission unavailable', false)
  }
}

enableButton.addEventListener('click', enableMotion)
snapButton.addEventListener('click', () => {
  lastSnapAt = performance.now()
  send({ type: 'snap' })
})

document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState === 'visible' && active && 'wakeLock' in navigator) {
    try {
      wakeLock = await navigator.wakeLock.request('screen')
    } catch {
    }
  }
})
