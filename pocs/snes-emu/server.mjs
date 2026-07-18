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
const databaseRoot = 'https://raw.githubusercontent.com/libretro/libretro-database/master/metadat'
const databaseName = 'Nintendo%20-%20Super%20Nintendo%20Entertainment%20System.dat'
const coverRoot = 'https://raw.githubusercontent.com/libretro-thumbnails/Nintendo_-_Super_Nintendo_Entertainment_System/master/Named_Boxarts'
const coverTreeUrl = 'https://api.github.com/repos/libretro-thumbnails/Nintendo_-_Super_Nintendo_Entertainment_System/git/trees/master?recursive=1'
let catalogPromise
let coverIndexPromise

const field = (body, name) => body.match(new RegExp(`${name}\\s+"([^"]+)"`))?.[1]

const entriesFrom = text => Array.from(text.matchAll(/game \(\n([\s\S]*?)\n\)(?:\n|$)/g), match => {
  const body = match[1]
  return {
    crc: body.match(/crc\s+([A-Fa-f0-9]{8})/)?.[1]?.toUpperCase(),
    title: field(body, 'name') || field(body, 'comment'),
    developer: field(body, 'developer'),
    year: field(body, 'releaseyear')
  }
}).filter(entry => entry.crc)

const titleKey = title => title
  .replace(/\.[^.]+$/, '')
  .replace(/\([^)]*\)|\[[^\]]*\]/g, ' ')
  .replace(/[^a-zA-Z0-9]+/g, ' ')
  .trim()
  .toLowerCase()

const compactTitleKey = title => titleKey(title).replaceAll(' ', '')

const titleScore = title => {
  let score = title.includes('(USA)') ? 4 : title.includes('(World)') ? 3 : title.includes('(Europe)') ? 2 : 1
  if (/\((Beta|Sample|Proto|Kiosk|Competition)/i.test(title)) score -= 20
  return score
}

const buildCatalog = async () => {
  const urls = [
    `${databaseRoot}/no-intro/${databaseName}`,
    `${databaseRoot}/developer/${databaseName}`,
    `${databaseRoot}/releaseyear/${databaseName}`
  ]
  const responses = await Promise.all(urls.map(url => fetch(url)))
  if (responses.some(response => !response.ok)) throw new Error('Catalog download failed')
  const [gamesText, developersText, yearsText] = await Promise.all(responses.map(response => response.text()))
  const developers = new Map(entriesFrom(developersText).map(entry => [entry.crc, entry.developer]))
  const years = new Map(entriesFrom(yearsText).map(entry => [entry.crc, entry.year]))
  const byCrc = new Map()
  const byTitle = new Map()
  const byCompactTitle = new Map()
  const scores = new Map()
  for (const game of entriesFrom(gamesText)) {
    const metadata = { title: game.title, developer: developers.get(game.crc), year: years.get(game.crc) }
    byCrc.set(game.crc, metadata)
    const key = titleKey(game.title)
    const score = titleScore(game.title)
    if (!byTitle.has(key) || score > scores.get(key)) {
      byTitle.set(key, metadata)
      scores.set(key, score)
    }
    const compactKey = compactTitleKey(game.title)
    if (!byCompactTitle.has(compactKey) || score > titleScore(byCompactTitle.get(compactKey).title)) byCompactTitle.set(compactKey, metadata)
  }
  return { byCrc, byTitle, byCompactTitle }
}

const fallbackCoverUrls = title => {
  if (!title) return []
  const safeTitle = title.replace(/[&*/:`<>?\\|]/g, '_')
  return Array.from(new Set([title, safeTitle])).map(name => `${coverRoot}/${encodeURIComponent(name)}.png`)
}

const coverNameKey = name => name.normalize('NFKD').replace('’', "'").replace(/\s+/g, ' ').trim().toLowerCase()

const regionFrom = title => title.match(/\((USA|Europe|Japan|World)/)?.[1]

const loadCoverIndex = async () => {
  const response = await fetch(coverTreeUrl, { headers: { Accept: 'application/vnd.github+json', 'User-Agent': 'SuperBlue' } })
  if (!response.ok) throw new Error('Cover index download failed')
  const data = await response.json()
  const names = data.tree
    .map(item => item.path)
    .filter(path => path.startsWith('Named_Boxarts/') && path.endsWith('.png'))
    .map(path => path.slice('Named_Boxarts/'.length, -4))
  const exact = new Map()
  const byTitle = new Map()
  for (const name of names) {
    exact.set(coverNameKey(name), name)
    const key = titleKey(name)
    const matches = byTitle.get(key) || []
    matches.push(name)
    byTitle.set(key, matches)
  }
  return { exact, byTitle }
}

const coverUrlsFor = async title => {
  if (!title) return []
  try {
    coverIndexPromise ||= loadCoverIndex()
    const index = await coverIndexPromise
    let name = index.exact.get(coverNameKey(title))
    if (!name) {
      const candidates = index.byTitle.get(titleKey(title)) || []
      const region = regionFrom(title)
      name = candidates.find(candidate => regionFrom(candidate) === region) || candidates[0]
    }
    return name ? [`${coverRoot}/${encodeURIComponent(name)}.png`] : []
  } catch (error) {
    process.stderr.write(`Cover index unavailable: ${error.message}\n`)
    coverIndexPromise = undefined
    return fallbackCoverUrls(title)
  }
}

const metadataFor = async url => {
  catalogPromise ||= buildCatalog()
  const catalog = await catalogPromise
  const crcCandidates = [url.searchParams.get('headerlessCrc'), url.searchParams.get('crc')].filter(Boolean).map(value => value.toUpperCase())
  let metadata
  for (const crc of crcCandidates) {
    metadata = catalog.byCrc.get(crc)
    if (metadata) break
  }
  let matchType = metadata ? 'crc' : ''
  const titleCandidates = [
    { title: url.searchParams.get('internalTitle'), type: 'internal' },
    { title: url.searchParams.get('title'), type: 'filename' }
  ].filter(candidate => candidate.title)
  for (const candidate of titleCandidates) {
    const { title } = candidate
    metadata ||= catalog.byTitle.get(titleKey(title))
    metadata ||= catalog.byCompactTitle.get(compactTitleKey(title))
    if (metadata && !matchType) matchType = candidate.type
  }
  metadata ||= { title: url.searchParams.get('title') || 'Unknown cartridge' }
  return { ...metadata, matched: Boolean(matchType), matchType: matchType || 'fallback', coverUrls: await coverUrlsFor(metadata.title) }
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

  const requestUrl = new URL(request.url, 'http://localhost')
  if (requestUrl.pathname === '/metadata') {
    try {
      const result = await metadataFor(requestUrl)
      const body = JSON.stringify(result)
      response.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'private, max-age=86400' })
      response.end(request.method === 'HEAD' ? '' : body)
    } catch {
      catalogPromise = undefined
      response.writeHead(503, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' })
      response.end('{"error":"Metadata unavailable"}')
    }
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
