const dgram = require('node:dgram')
const http = require('node:http')

const expected = { type: 'motion', ax: 0.17, ay: -0.23, az: 0.94 }
const httpPort = Number(process.env.GAME_PORT || 8080)
const udpPort = Number(process.env.UDP_PORT || 5005)
const timeout = setTimeout(() => {
  process.stderr.write('UDP bridge event was not received\n')
  process.exit(1)
}, 2000)

const request = http.get(`http://127.0.0.1:${httpPort}/events`, (response) => {
  let content = ''
  response.on('data', (chunk) => {
    content += chunk.toString()
    const match = content.match(/data: (\{[^\n]+\})/)
    if (!match) return
    const packet = JSON.parse(match[1])
    if (packet.type !== expected.type || packet.ax !== expected.ax || packet.ay !== expected.ay || packet.az !== expected.az) {
      process.stderr.write('UDP bridge changed the motion values\n')
      process.exit(1)
    }
    clearTimeout(timeout)
    request.destroy()
    process.stdout.write('UDP to browser event bridge passed\n')
  })
  const udp = dgram.createSocket('udp4')
  udp.send(JSON.stringify(expected), udpPort, '127.0.0.1', () => udp.close())
})

request.on('error', (error) => {
  if (error.code === 'ECONNRESET') return
  process.stderr.write(`${error.message}\n`)
  process.exit(1)
})
