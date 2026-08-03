const dgram = require('node:dgram')
const fs = require('node:fs')
const net = require('node:net')
const os = require('node:os')
const path = require('node:path')
const { spawn } = require('node:child_process')

const tcpBlocker = net.createServer()
const udpBlocker = dgram.createSocket('udp4')
const controllerBlocker = net.createServer()
const statePath = path.join(os.tmpdir(), `fly-catcher-ports-${process.pid}.state`)
const lanHost = Object.values(os.networkInterfaces()).flat().find((entry) => entry && (entry.family === 'IPv4' || entry.family === 4) && !entry.internal && (/^10\./.test(entry.address) || /^192\.168\./.test(entry.address) || /^172\.(1[6-9]|2\d|3[01])\./.test(entry.address)))?.address
let child
let output = ''

function closeBlockers() {
  tcpBlocker.close()
  udpBlocker.close()
  controllerBlocker.close()
}

function fail(message) {
  if (child) child.kill('SIGTERM')
  closeBlockers()
  try {
    fs.unlinkSync(statePath)
  } catch {
  }
  process.stderr.write(`${message}\n${output}`)
  process.exit(1)
}

tcpBlocker.on('error', (error) => fail(error.message))
udpBlocker.on('error', (error) => fail(error.message))
controllerBlocker.on('error', (error) => fail(error.message))

tcpBlocker.listen(0, '127.0.0.1', () => {
  udpBlocker.bind(0, '0.0.0.0', () => {
    controllerBlocker.listen(0, lanHost, () => {
      const preferredHttpPort = tcpBlocker.address().port
      const preferredUdpPort = udpBlocker.address().port
      const preferredControllerPort = controllerBlocker.address().port
      child = spawn(process.execPath, ['server.js'], {
        cwd: path.join(__dirname, '..'),
        env: {
          ...process.env,
          CONTROLLER_HOST: lanHost,
          CONTROLLER_PORT: String(preferredControllerPort),
          GAME_PORT: String(preferredHttpPort),
          UDP_PORT: String(preferredUdpPort),
          STATE_FILE: statePath
        },
        stdio: ['ignore', 'pipe', 'pipe']
      })
      child.stdout.on('data', (data) => { output += data })
      child.stderr.on('data', (data) => { output += data })
      child.on('exit', (code) => {
        if (code && code !== 0) fail(`Fallback server exited with ${code}`)
      })

      const timeout = setTimeout(() => fail('Fallback ports were not selected'), 5000)
      const check = setInterval(() => {
        if (!fs.existsSync(statePath)) return
        const saved = JSON.parse(fs.readFileSync(statePath, 'utf8'))
        if (saved.httpPort === preferredHttpPort || saved.udpPort === preferredUdpPort || saved.controllerPort === preferredControllerPort) fail('An occupied port was selected')
        clearInterval(check)
        clearTimeout(timeout)
        child.once('exit', () => {
          closeBlockers()
          if (fs.existsSync(statePath)) fail('Fallback state was not removed')
          process.stdout.write('Occupied port fallback and state cleanup passed\n')
        })
        child.kill('SIGTERM')
      }, 25)
    })
  })
})
