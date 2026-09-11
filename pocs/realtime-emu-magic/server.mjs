import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { agentStatus, callCodex } from './agent.mjs'
import { validateRequest, validateReply, maxRequestBytes } from './public/contracts.js'

const files = new Map([
  ['/', ['index.html', 'text/html']], ['/styles.css', ['styles.css', 'text/css']],
  ['/app.js', ['app.js', 'text/javascript']], ['/player.html', ['player.html', 'text/html']],
  ['/player.js', ['player.js', 'text/javascript']], ['/contracts.js', ['contracts.js', 'text/javascript']],
  ['/memory.js', ['memory.js', 'text/javascript']], ['/patch-engine.js', ['patch-engine.js', 'text/javascript']]
])

export function createApp({ agent = callCodex, status = agentStatus } = {}) {
  let busy = false
  let activeController
  const server = createServer(async (request, response) => {
    const send = (code, body) => {
      if (response.destroyed) return
      response.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' })
      response.end(JSON.stringify(body))
    }
    response.setHeader('X-Content-Type-Options', 'nosniff')
    response.setHeader('Referrer-Policy', 'no-referrer')
    response.setHeader('X-Frame-Options', 'SAMEORIGIN')
    const host = request.headers.host
    if (!/^(127\.0\.0\.1|localhost):\d+$/.test(host || '')) return send(403, { error: 'Local access only.' })
    const origin = `http://${host}`
    if (request.headers.origin && request.headers.origin !== origin) return send(403, { error: 'Cross-origin access denied.' })
    let url
    try { url = new URL(request.url, origin) } catch { return send(400, { error: 'Invalid request URL.' }) }
    try {
      if (request.method === 'GET' && url.pathname === '/health') return send(200, { app: 'realtime-emu-magic', ok: true })
      if (request.method === 'GET' && url.pathname === '/api/status') return send(200, { ...await status(), busy })
      if (request.method === 'POST' && url.pathname === '/api/change') {
        if (request.headers.origin !== origin || request.headers['content-type'] !== 'application/json') return send(403, { error: 'Send same-origin JSON requests.' })
        if (busy) return send(409, { error: 'Codex is already working on a request.' })
        let body = ''
        for await (const chunk of request) {
          body += chunk.toString()
          if (Buffer.byteLength(body) > maxRequestBytes) return send(413, { error: 'Request too large.' })
        }
        let input
        try { input = validateRequest(JSON.parse(body)) } catch (error) { return send(400, { error: error.message }) }
        if (busy) return send(409, { error: 'Codex is already working on a request.' })
        busy = true
        const controller = new AbortController()
        activeController = controller
        response.once('close', () => controller.abort())
        try {
          send(200, validateReply(await agent(input, controller.signal, input.model)))
        } catch (error) {
          send(502, { error: error.message })
        } finally {
          busy = false
          activeController = null
        }
        return
      }
      if (request.method !== 'GET' || !files.has(url.pathname)) return send(404, { error: 'Not found.' })
      const [file, type] = files.get(url.pathname)
      const content = await readFile(new URL(`./public/${file}`, import.meta.url))
      response.writeHead(200, { 'Content-Type': `${type}; charset=utf-8`, 'Cache-Control': 'no-cache' })
      response.end(content)
    } catch {
      send(500, { error: 'The server could not complete this request.' })
    }
  })
  server.on('shutdown', () => activeController?.abort())
  return server
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const config = await readFile(new URL('./scripts/ports.env', import.meta.url), 'utf8')
  const port = Number(process.env.PORT || config.match(/^APP=(\d+)$/m)?.[1])
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid APP port.')
  const server = createApp()
  server.listen(port, '127.0.0.1', () => process.stdout.write(`Realtime Emu Magic running at http://127.0.0.1:${port}\n`))
  process.on('SIGTERM', () => {
    server.emit('shutdown')
    server.close(() => process.exit(0))
  })
}
