const dgram = require('node:dgram')
const crypto = require('node:crypto')
const { execFileSync } = require('node:child_process')
const fs = require('node:fs')
const http = require('node:http')
const https = require('node:https')
const os = require('node:os')
const path = require('node:path')

const httpHost = '127.0.0.1'
const preferredHttpPort = Number(process.env.GAME_PORT || 8080)
const preferredUdpPort = Number(process.env.UDP_PORT || 5005)
const preferredControllerPort = Number(process.env.CONTROLLER_PORT || 8443)
const udpHost = '0.0.0.0'
const publicDir = path.join(__dirname, 'public')
const certDir = path.join(__dirname, '.certs')
const statePath = path.resolve(process.env.STATE_FILE || path.join(__dirname, '.fly-catcher.state'))
const clients = new Set()
let httpPort = preferredHttpPort
let udpPort = preferredUdpPort
let udp
let pairingServer
let controllerServer
let relay
let pairingUrl = null
let controllerUrl = null
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

const controllerFiles = new Map([
  ['/controller', ['controller.html', 'text/html; charset=utf-8']],
  ['/controller.css', ['controller.css', 'text/css; charset=utf-8']],
  ['/controller.js', ['controller.js', 'text/javascript; charset=utf-8']]
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
const controllerToken = crypto.randomBytes(16).toString('hex')

function ensureCertificates(host) {
  fs.mkdirSync(certDir, { recursive: true })
  const caKey = path.join(certDir, 'ca.key')
  const caCert = path.join(certDir, 'ca.crt')
  const caDer = path.join(certDir, 'ca.cer')
  const caSerial = path.join(certDir, 'ca.srl')
  const serverKey = path.join(certDir, 'server.key')
  const serverCert = path.join(certDir, 'server.crt')
  const serverRequest = path.join(certDir, 'server.csr')
  const serverExtensions = path.join(certDir, 'server.ext')
  const hostFile = path.join(certDir, 'host')
  if (!fs.existsSync(caKey) || !fs.existsSync(caCert)) {
    execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-sha256', '-days', '3650', '-nodes', '-keyout', caKey, '-out', caCert, '-subj', '/CN=Fly Catcher Local Root', '-addext', 'basicConstraints=critical,CA:TRUE', '-addext', 'keyUsage=critical,keyCertSign,cRLSign', '-addext', 'subjectKeyIdentifier=hash'], { stdio: 'ignore' })
  }
  const savedHost = fs.existsSync(hostFile) ? fs.readFileSync(hostFile, 'utf8').trim() : ''
  if (savedHost !== host || !fs.existsSync(serverKey) || !fs.existsSync(serverCert)) {
    fs.writeFileSync(serverExtensions, `basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=IP:${host}\nauthorityKeyIdentifier=keyid,issuer\n`)
    execFileSync('openssl', ['req', '-new', '-newkey', 'rsa:2048', '-sha256', '-nodes', '-keyout', serverKey, '-out', serverRequest, '-subj', `/CN=${host}`], { stdio: 'ignore' })
    execFileSync('openssl', ['x509', '-req', '-in', serverRequest, '-CA', caCert, '-CAkey', caKey, '-CAserial', caSerial, '-CAcreateserial', '-out', serverCert, '-days', '365', '-sha256', '-extfile', serverExtensions], { stdio: 'ignore' })
    fs.writeFileSync(hostFile, host)
    fs.unlinkSync(serverRequest)
  }
  execFileSync('openssl', ['x509', '-in', caCert, '-outform', 'der', '-out', caDer], { stdio: 'ignore' })
  return { caCert, caDer, serverKey, serverCert }
}

const certificates = controllerHost ? ensureCertificates(controllerHost) : null

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
      controllerUrl,
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
  if (relay) relay.close()
  if (pairingServer) pairingServer.close()
  if (controllerServer) controllerServer.close()
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
  const certificatePath = '/fly-catcher-ca.mobileconfig'
  const healthUri = controllerUri.replace('/controller?', '/health.svg?')
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#17131e">
<title>Fly Catcher Pairing</title>
<style>
*{box-sizing:border-box}html,body{min-height:100%;margin:0}body{padding:18px;display:grid;place-items:center;color:#fff0c9;font-family:Rockwell,"Courier New",monospace;background:#17131e}.panel,.checking{width:min(470px,100%);padding:25px;border:5px solid #09070d;background:#2e2336;box-shadow:9px 9px 0 #ff5b4d}.checking{text-align:center}.pulse{width:42px;height:42px;margin:0 auto 18px;background:#ffd447;border:7px solid #ff5b4d;box-shadow:5px 5px 0 #09070d;animation:pulse .7s steps(2,end) infinite}small{color:#ffd447;font-weight:900;letter-spacing:.16em;text-transform:uppercase}h1{margin:8px 0 12px;font-family:Impact,Haettenschweiler,fantasy;font-size:42px;line-height:.95;text-transform:uppercase;text-shadow:4px 4px 0 #ff5b4d}p{line-height:1.4}.address{padding:10px;color:#17131e;font-weight:900;background:#ffd447}.step{margin:10px 0;padding:10px;border-left:5px solid #ff5b4d;background:#17131e}.step strong{color:#ffd447}a{margin-top:12px;padding:15px;display:block;color:#17131e;font-weight:900;text-align:center;text-decoration:none;text-transform:uppercase;background:#65d6a7;box-shadow:5px 5px 0 #09070d}a.cert{background:#ffd447}a:active{transform:translate(3px,3px);box-shadow:2px 2px 0 #09070d}[hidden]{display:none}@keyframes pulse{50%{transform:scale(.72);background:#65d6a7}}
</style>
</head>
<body>
<section class="checking" id="checking">
<div class="pulse"></div>
<small>Browser controller</small>
<h1>Checking local trust</h1>
<p>Opening the controller automatically when this phone is ready.</p>
</section>
<main class="panel" id="setup" hidden>
<small>Browser controller</small>
<h1>Pair this phone</h1>
<p class="address">${controllerHost}:${udpPort}</p>
<p class="step"><strong>1.</strong> Install the local certificate once.</p>
<a class="cert" href="${certificatePath}">Download certificate</a>
<p class="step"><strong>2.</strong> In Settings, install the downloaded profile. Then enable full trust under General, About, Certificate Trust Settings.</p>
<p class="step"><strong>3.</strong> Return here and open the controller.</p>
<a href="${controllerUri}">Open browser controller</a>
</main>
<script>
const target=${JSON.stringify(controllerUri)}
const checking=document.getElementById('checking')
const setup=document.getElementById('setup')
let finished=false
const reveal=()=>{if(finished)return;finished=true;checking.hidden=true;setup.hidden=false}
const probe=new Image()
probe.onload=()=>{finished=true;location.replace(target)}
probe.onerror=reveal
probe.src=${JSON.stringify(healthUri)}+'&time='+Date.now()
setTimeout(reveal,1800)
</script>
</body>
</html>`
}

function certificateProfile() {
  const data = fs.readFileSync(certificates.caDer).toString('base64').match(/.{1,64}/g).join('\n')
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
<key>PayloadContent</key>
<array>
<dict>
<key>PayloadCertificateFileName</key>
<string>fly-catcher-local-root.cer</string>
<key>PayloadContent</key>
<data>${data}</data>
<key>PayloadDescription</key>
<string>Trusts the local Fly Catcher browser controller.</string>
<key>PayloadDisplayName</key>
<string>Fly Catcher Local Root</string>
<key>PayloadIdentifier</key>
<string>local.flycatcher.root</string>
<key>PayloadType</key>
<string>com.apple.security.root</string>
<key>PayloadUUID</key>
<string>52EE75E9-A52A-45B6-B75F-2649CBF50FA2</string>
<key>PayloadVersion</key>
<integer>1</integer>
</dict>
</array>
<key>PayloadDescription</key>
<string>Enables the private HTTPS Fly Catcher controller.</string>
<key>PayloadDisplayName</key>
<string>Fly Catcher Local Certificate</string>
<key>PayloadIdentifier</key>
<string>local.flycatcher.profile</string>
<key>PayloadOrganization</key>
<string>Fly Catcher</string>
<key>PayloadRemovalDisallowed</key>
<false/>
<key>PayloadType</key>
<string>Configuration</string>
<key>PayloadUUID</key>
<string>C46303AA-BE7E-4182-BA05-0FDE0447F66A</string>
<key>PayloadVersion</key>
<integer>1</integer>
</dict>
</plist>`
}

function finishStartup(pairPort, controllerPort) {
  pairingUrl = pairPort ? `http://${controllerHost}:${pairPort}/pair` : null
  const saved = { pid: process.pid, httpPort, udpPort, controllerHost, controllerPort, controllerUrl, pairPort, pairingUrl, controllerToken }
  const temporaryStatePath = `${statePath}.${process.pid}`
  fs.writeFileSync(temporaryStatePath, JSON.stringify(saved))
  fs.renameSync(temporaryStatePath, statePath)
  process.stdout.write(`Fly Catcher: http://${httpHost}:${httpPort}\n`)
  process.stdout.write(`iPhone UDP receiver: ${udpHost}:${udpPort}\n`)
  process.stdout.write(`Phone pairing page: ${pairingUrl || 'Private IPv4 address unavailable'}\n`)
  if (controllerUrl) process.stdout.write(`Browser controller: ${controllerUrl}\n`)
  if (httpPort !== preferredHttpPort) process.stdout.write(`HTTP port ${preferredHttpPort} was occupied; using ${httpPort}\n`)
  if (udpPort !== preferredUdpPort) process.stdout.write(`UDP port ${preferredUdpPort} was occupied; using ${udpPort}\n`)
}

function listenPairing(port, attempt = 0) {
  const candidate = http.createServer((request, response) => {
    response.setHeader('Cache-Control', 'no-store')
    response.setHeader('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src https:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
    response.setHeader('X-Content-Type-Options', 'nosniff')
    response.setHeader('X-Frame-Options', 'DENY')
    const pathname = request.url?.split('?')[0]
    if (pathname === '/fly-catcher-ca.mobileconfig') {
      response.setHeader('Content-Type', 'application/x-apple-aspen-config')
      response.setHeader('Content-Disposition', 'attachment; filename="fly-catcher-ca.mobileconfig"')
      response.end(certificateProfile())
      return
    }
    if (pathname !== '/pair') {
      response.writeHead(404)
      response.end('Not found')
      return
    }
    response.setHeader('Content-Type', 'text/html; charset=utf-8')
    response.end(pairingPage(`${controllerUrl}?token=${controllerToken}`))
  })
  candidate.once('error', (error) => {
    if (error.code === 'EADDRINUSE' && attempt < 100) {
      listenPairing(port + 1, attempt + 1)
      return
    }
    process.stderr.write(`Pairing HTTP error: ${error.message}\n`)
    process.exit(1)
  })
  candidate.listen(port, controllerHost, () => {
    candidate.removeAllListeners('error')
    candidate.on('error', (error) => process.stderr.write(`Pairing HTTP error: ${error.message}\n`))
    pairingServer = candidate
    finishStartup(port, Number(new URL(controllerUrl).port))
  })
}

function relayBrowserPacket(input, remoteAddress, response) {
  if (!isPrivateAddress(remoteAddress)) {
    response.writeHead(403)
    response.end()
    return
  }
  const packet = normalizePacket(input)
  if (!packet) {
    response.writeHead(400)
    response.end()
    return
  }
  relay.send(JSON.stringify(packet), udpPort, '127.0.0.1', (error) => {
    response.writeHead(error ? 500 : 204)
    response.end()
  })
}

function controllerRequest(request, response) {
  response.setHeader('Cache-Control', 'no-store')
  response.setHeader('Content-Security-Policy', "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
  response.setHeader('X-Content-Type-Options', 'nosniff')
  response.setHeader('X-Frame-Options', 'DENY')
  const url = new URL(request.url, `https://${controllerHost}`)
  if (request.method === 'GET' && url.pathname === '/health.svg') {
    if (url.searchParams.get('token') !== controllerToken) {
      response.writeHead(403)
      response.end()
      return
    }
    response.setHeader('Content-Type', 'image/svg+xml')
    response.end('<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><path fill="#65d6a7" d="M0 0h1v1H0z"/></svg>')
    return
  }
  if (request.method === 'POST' && url.pathname === '/api/control') {
    if (request.headers['x-fly-token'] !== controllerToken) {
      response.writeHead(403)
      response.end()
      return
    }
    let body = ''
    request.on('data', (chunk) => {
      body += chunk
      if (body.length > 1024) request.destroy()
    })
    request.on('end', () => {
      let input
      try {
        input = JSON.parse(body)
      } catch {
        response.writeHead(400)
        response.end()
        return
      }
      relayBrowserPacket(input, request.socket.remoteAddress, response)
    })
    return
  }
  if (url.pathname === '/controller' && url.searchParams.get('token') !== controllerToken) {
    response.writeHead(403)
    response.end('Pairing link required')
    return
  }
  const file = controllerFiles.get(url.pathname)
  if (!file) {
    response.writeHead(404)
    response.end('Not found')
    return
  }
  fs.readFile(path.join(publicDir, file[0]), (error, content) => {
    if (error) {
      response.writeHead(500)
      response.end('Unable to load controller')
      return
    }
    response.setHeader('Content-Type', file[1])
    response.end(content)
  })
}

function listenController(port, attempt = 0) {
  const candidate = https.createServer({ key: fs.readFileSync(certificates.serverKey), cert: fs.readFileSync(certificates.serverCert) }, controllerRequest)
  candidate.once('error', (error) => {
    if (error.code === 'EADDRINUSE' && attempt < 100) {
      listenController(port + 1, attempt + 1)
      return
    }
    process.stderr.write(`Controller HTTPS error: ${error.message}\n`)
    process.exit(1)
  })
  candidate.listen(port, controllerHost, () => {
    candidate.removeAllListeners('error')
    candidate.on('error', (error) => process.stderr.write(`Controller HTTPS error: ${error.message}\n`))
    controllerServer = candidate
    controllerUrl = `https://${controllerHost}:${port}/controller`
    listenPairing(httpPort)
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
    if (certificates) listenController(preferredControllerPort)
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
    relay = dgram.createSocket('udp4')
    udpPort = port
    listenHttp(preferredHttpPort)
  })
}

bindUdp(preferredUdpPort)
