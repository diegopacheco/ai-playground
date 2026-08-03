const dgram = require('node:dgram')
const fs = require('node:fs')
const http = require('node:http')
const os = require('node:os')
const path = require('node:path')

const httpHost = '127.0.0.1'
const preferredHttpPort = Number(process.env.GAME_PORT || 8080)
const preferredUdpPort = Number(process.env.UDP_PORT || 5005)
const udpHost = '0.0.0.0'
const publicDir = path.join(__dirname, 'public')
const statePath = path.resolve(process.env.STATE_FILE || path.join(__dirname, '.fly-catcher.state'))
const clients = new Set()
let httpPort = preferredHttpPort
let udpPort = preferredUdpPort
let udp
let pairingServer
let pairingUrl = null
let packetCount = 0
let lastPacketAt = null
let lastControllerAddress = null

const files = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/index.html', ['index.html', 'text/html; charset=utf-8']],
  ['/styles.css', ['styles.css', 'text/css; charset=utf-8']],
  ['/qr.js', ['qr.js', 'text/javascript; charset=utf-8']],
  ['/game.js', ['game.js', 'text/javascript; charset=utf-8']]
])

function isPrivateAddress(address) {
  if (address.startsWith('::ffff:')) return isPrivateAddress(address.slice(7))
  if (address === '::1' || address.startsWith('fe80:') || address.startsWith('fc') || address.startsWith('fd')) return true
  const parts = address.split('.').map(Number)
  if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) return false
  return parts[0] === 10 || parts[0] === 127 || (parts[0] === 169 && parts[1] === 254) || (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) || (parts[0] === 192 && parts[1] === 168)
}

function findControllerHost() {
  if (process.env.CONTROLLER_HOST) {
    const configured = process.env.CONTROLLER_HOST.trim()
    return isPrivateAddress(configured) && !configured.startsWith('127.') && !configured.startsWith('169.254.') ? configured : null
  }
  const addresses = []
  for (const [name, entries] of Object.entries(os.networkInterfaces())) {
    for (const entry of entries || []) {
      if ((entry.family === 'IPv4' || entry.family === 4) && !entry.internal && isPrivateAddress(entry.address) && !entry.address.startsWith('169.254.')) {
        addresses.push({ name, address: entry.address })
      }
    }
  }
  addresses.sort((left, right) => Number(right.name === 'en0') - Number(left.name === 'en0'))
  return addresses[0]?.address || null
}

const controllerHost = findControllerHost()

function normalizePacket(input) {
  if (!input || typeof input !== 'object') return null
  if (input.type === 'snap') return { type: 'snap', at: Date.now() }
  if (input.type !== 'motion') return null
  const ax = Number(input.ax)
  const ay = Number(input.ay)
  const az = Number(input.az)
  if (![ax, ay, az].every(Number.isFinite)) return null
  return {
    type: 'motion',
    ax: Math.max(-4, Math.min(4, ax)),
    ay: Math.max(-4, Math.min(4, ay)),
    az: Math.max(-4, Math.min(4, az)),
    at: Date.now()
  }
}

function broadcast(packet) {
  const payload = `data: ${JSON.stringify(packet)}\n\n`
  for (const client of clients) client.write(payload)
}

function receivePacket(message, remote) {
  if (!isPrivateAddress(remote.address) || message.length > 1024) return
  let input
  try {
    input = JSON.parse(message.toString('utf8'))
  } catch {
    return
  }
  const packet = normalizePacket(input)
  if (!packet) return
  packetCount += 1
  lastPacketAt = packet.at
  lastControllerAddress = remote.address
  broadcast(packet)
}

const server = http.createServer((request, response) => {
  response.setHeader('Cache-Control', 'no-store')
  response.setHeader('Content-Security-Policy', "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
  response.setHeader('X-Content-Type-Options', 'nosniff')
  response.setHeader('X-Frame-Options', 'DENY')

  if (request.url === '/events') {
    response.writeHead(200, {
      'Content-Type': 'text/event-stream',
      Connection: 'keep-alive'
    })
    response.write('retry: 1000\n\n')
    clients.add(response)
    request.on('close', () => clients.delete(response))
    return
  }

  if (request.url === '/api/status') {
    response.setHeader('Content-Type', 'application/json; charset=utf-8')
    response.end(JSON.stringify({
      running: true,
      http: `${httpHost}:${httpPort}`,
      udp: `${udpHost}:${udpPort}`,
      controllerHost,
      controllerUri: controllerHost ? `flycatcher://connect?host=${encodeURIComponent(controllerHost)}&port=${udpPort}` : null,
      pairingUrl,
      packetCount,
      lastPacketAt,
      lastControllerAddress,
      browserClients: clients.size
    }))
    return
  }

  const file = files.get(request.url)
  if (!file) {
    response.writeHead(404)
    response.end('Not found')
    return
  }

  fs.readFile(path.join(publicDir, file[0]), (error, content) => {
    if (error) {
      response.writeHead(500)
      response.end('Unable to load game')
      return
    }
    response.setHeader('Content-Type', file[1])
    response.end(content)
  })
})

function shutdown() {
  for (const client of clients) client.end()
  if (udp) udp.close()
  if (pairingServer) pairingServer.close()
  try {
    const saved = JSON.parse(fs.readFileSync(statePath, 'utf8'))
    if (saved.pid === process.pid) fs.unlinkSync(statePath)
  } catch {
  }
  server.close(() => process.exit(0))
}

process.on('SIGINT', shutdown)
process.on('SIGTERM', shutdown)

function pairingPage(controllerUri) {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#17131e">
<title>Fly Catcher Pairing</title>
<style>
*{box-sizing:border-box}html,body{height:100%;overflow:hidden}body{margin:0;padding:18px;display:grid;place-items:center;color:#fff0c9;font-family:Rockwell,"Courier New",monospace;background:#17131e}.panel{width:min(430px,100%);padding:28px;border:5px solid #09070d;background:#2e2336;box-shadow:9px 9px 0 #ff5b4d}small{color:#ffd447;font-weight:900;letter-spacing:.16em;text-transform:uppercase}h1{margin:8px 0 12px;font-family:Impact,Haettenschweiler,fantasy;font-size:42px;line-height:.95;text-transform:uppercase;text-shadow:4px 4px 0 #ff5b4d}p{line-height:1.45}.address{padding:11px;color:#17131e;font-weight:900;background:#ffd447}.help{font-size:12px;color:#d8aeb1}a{margin-top:18px;padding:17px;display:block;color:#17131e;font-weight:900;text-align:center;text-decoration:none;text-transform:uppercase;background:#65d6a7;box-shadow:5px 5px 0 #09070d}a:active{transform:translate(3px,3px);box-shadow:2px 2px 0 #09070d}
</style>
</head>
<body>
<main class="panel">
<small>Controller 1.1 required</small>
<h1>Install then pair</h1>
<p>Safari cannot send UDP. Install the iPhone controller from Xcode on the Mac, then open it here.</p>
<p class="address">${controllerHost}:${udpPort}</p>
<a href="${controllerUri}">Open installed controller</a>
<p class="help">If Safari reports an invalid address, Controller 1.1 is not installed on this iPhone.</p>
</main>
</body>
</html>`
}

function finishStartup(pairPort, controllerUri) {
  pairingUrl = pairPort ? `http://${controllerHost}:${pairPort}/pair` : null
  const saved = { pid: process.pid, httpPort, udpPort, controllerHost, controllerUri, pairPort, pairingUrl }
  const temporaryStatePath = `${statePath}.${process.pid}`
  fs.writeFileSync(temporaryStatePath, JSON.stringify(saved))
  fs.renameSync(temporaryStatePath, statePath)
  process.stdout.write(`Fly Catcher: http://${httpHost}:${httpPort}\n`)
  process.stdout.write(`iPhone UDP receiver: ${udpHost}:${udpPort}\n`)
  process.stdout.write(`Phone pairing page: ${pairingUrl || 'Private IPv4 address unavailable'}\n`)
  if (httpPort !== preferredHttpPort) process.stdout.write(`HTTP port ${preferredHttpPort} was occupied; using ${httpPort}\n`)
  if (udpPort !== preferredUdpPort) process.stdout.write(`UDP port ${preferredUdpPort} was occupied; using ${udpPort}\n`)
}

function listenPairing(port, controllerUri, attempt = 0) {
  const candidate = http.createServer((request, response) => {
    response.setHeader('Cache-Control', 'no-store')
    response.setHeader('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
    response.setHeader('X-Content-Type-Options', 'nosniff')
    response.setHeader('X-Frame-Options', 'DENY')
    if (request.url?.split('?')[0] !== '/pair') {
      response.writeHead(404)
      response.end('Not found')
      return
    }
    response.setHeader('Content-Type', 'text/html; charset=utf-8')
    response.end(pairingPage(controllerUri))
  })
  candidate.once('error', (error) => {
    if (error.code === 'EADDRINUSE' && attempt < 100) {
      listenPairing(port + 1, controllerUri, attempt + 1)
      return
    }
    process.stderr.write(`Pairing HTTP error: ${error.message}\n`)
    process.exit(1)
  })
  candidate.listen(port, controllerHost, () => {
    candidate.removeAllListeners('error')
    candidate.on('error', (error) => process.stderr.write(`Pairing HTTP error: ${error.message}\n`))
    pairingServer = candidate
    finishStartup(port, controllerUri)
  })
}

function listenHttp(port, attempt = 0) {
  const onError = (error) => {
    if (error.code === 'EADDRINUSE' && attempt < 100) {
      listenHttp(port + 1, attempt + 1)
      return
    }
    process.stderr.write(`HTTP error: ${error.message}\n`)
    process.exit(1)
  }
  server.once('error', onError)
  server.listen(port, httpHost, () => {
    server.removeListener('error', onError)
    server.on('error', (error) => process.stderr.write(`HTTP error: ${error.message}\n`))
    httpPort = port
    const controllerUri = controllerHost ? `flycatcher://connect?host=${encodeURIComponent(controllerHost)}&port=${udpPort}` : null
    if (controllerUri) listenPairing(httpPort, controllerUri)
    else finishStartup(null, null)
  })
}

function bindUdp(port, attempt = 0) {
  const socket = dgram.createSocket('udp4')
  const onError = (error) => {
    if (error.code === 'EADDRINUSE' && attempt < 100) {
      bindUdp(port + 1, attempt + 1)
      return
    }
    process.stderr.write(`UDP error: ${error.message}\n`)
    process.exit(1)
  }
  socket.once('error', onError)
  socket.bind(port, udpHost, () => {
    socket.removeListener('error', onError)
    socket.on('error', (error) => process.stderr.write(`UDP error: ${error.message}\n`))
    socket.on('message', receivePacket)
    udp = socket
    udpPort = port
    listenHttp(preferredHttpPort)
  })
}

bindUdp(preferredUdpPort)
