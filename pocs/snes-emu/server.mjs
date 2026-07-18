import { createReadStream } from 'node:fs'
import { stat } from 'node:fs/promises'
import { createServer } from 'node:http'
import { extname, join, normalize, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('.', import.meta.url)).replace(/[\\/]$/, '')
const port = Number(process.argv[2] || process.env.PORT || 9011)
const publicFiles = new Set(['index.html', 'player.html', 'styles.css', 'app.js', 'player.js'])
const types = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml'
}

const server = createServer(async (request, response) => {
  if (!['GET', 'HEAD'].includes(request.method)) {
    response.writeHead(405, { Allow: 'GET, HEAD' })
    response.end()
    return
  }

  if (request.url === '/health') {
    response.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' })
    response.end('{"status":"ok"}')
    return
  }

  try {
    const requestPath = decodeURIComponent(new URL(request.url, 'http://localhost').pathname)
    const relativePath = requestPath === '/' ? 'index.html' : normalize(requestPath).replace(/^[/\\]+/, '')
    if (!publicFiles.has(relativePath)) throw new Error('Not public')
    const filePath = join(root, relativePath)
    if (!filePath.startsWith(`${root}${sep}`)) throw new Error('Invalid path')
    const fileStat = await stat(filePath)
    if (!fileStat.isFile()) throw new Error('Not a file')
    response.writeHead(200, {
      'Content-Type': types[extname(filePath)] || 'application/octet-stream',
      'Content-Length': fileStat.size,
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff'
    })
    if (request.method === 'HEAD') response.end()
    else createReadStream(filePath).pipe(response)
  } catch {
    response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' })
    response.end('Not found')
  }
})

server.listen(port, '127.0.0.1', () => process.stdout.write(`SuperBlue running at http://127.0.0.1:${port}\n`))

const close = () => server.close(() => process.exit(0))
process.on('SIGINT', close)
process.on('SIGTERM', close)
