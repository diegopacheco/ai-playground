const http = require("node:http")
const fs = require("node:fs")
const path = require("node:path")
const crypto = require("node:crypto")

const port = Number(process.env.PORT || 4177)
const publicDir = path.join(__dirname, "public")
const clients = new Set()
const players = new Map()
const columns = 21
const rows = 15
const tile = 50
const boardX = 525
const boardY = 360
const levelCount = 9
let level = 1
let loop = 1
let transitioning = false
const pieces = Array.from({ length: columns * rows }, (_, id) => ({
  id,
  column: id % columns,
  row: Math.floor(id / columns),
  x: null,
  y: null,
  placed: false,
  owner: null
}))

const names = [
  "Amber Badger",
  "Blue Heron",
  "Coral Fox",
  "Dapper Owl",
  "Fern Rabbit",
  "Golden Moth",
  "Indigo Finch",
  "Jolly Otter",
  "Kind Magpie",
  "Lemon Panda",
  "Merry Gecko",
  "Nimble Lynx",
  "Peach Koala",
  "Quiet Quail",
  "Ruby Crab",
  "Sunny Newt",
  "Teal Tiger",
  "Velvet Yak"
]

const colors = [
  "#0b6e69",
  "#1769aa",
  "#d7503f",
  "#7452a3",
  "#d17a00",
  "#2f7d32",
  "#bd3d74",
  "#5d63d8"
]

const types = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml"
}

function sendJson(response, status, data) {
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store"
  })
  response.end(JSON.stringify(data))
}

function broadcast(data, excludedId) {
  const payload = `data: ${JSON.stringify(data)}\n\n`
  for (const client of clients) {
    if (client.id !== excludedId) client.response.write(payload)
  }
}

function publicPlayer(player) {
  return {
    id: player.id,
    name: player.name,
    color: player.color,
    x: player.x,
    y: player.y,
    placed: player.placed
  }
}

function progress() {
  return pieces.reduce((total, piece) => total + Number(piece.placed), 0)
}

function advanceLevel() {
  if (transitioning) return
  transitioning = true
  const nextLevel = level === levelCount ? 1 : level + 1
  const nextLoop = nextLevel === 1 ? loop + 1 : loop
  broadcast({ type: "complete", level, loop, nextLevel, nextLoop })
  setTimeout(() => {
    level = nextLevel
    loop = nextLoop
    for (const piece of pieces) {
      piece.x = null
      piece.y = null
      piece.placed = false
      piece.owner = null
    }
    transitioning = false
    broadcast({ type: "level", level, loop, pieces, progress: 0 })
  }, 1800)
}

function playerFor(id) {
  return players.get(String(id || ""))
}

function join(id, requestedName) {
  const cleanId = String(id || crypto.randomUUID()).slice(0, 80)
  const existing = players.get(cleanId)
  if (existing) {
    existing.lastSeen = Date.now()
    return existing
  }
  const cleanName = String(requestedName || names[players.size % names.length])
    .replace(/[^\p{L}\p{N} _-]/gu, "")
    .trim()
    .slice(0, 28)
  const player = {
    id: cleanId,
    name: cleanName || names[players.size % names.length],
    color: colors[players.size % colors.length],
    x: boardX + 520,
    y: boardY + 375,
    placed: 0,
    lastSeen: Date.now()
  }
  players.set(cleanId, player)
  broadcast({ type: "player", player: publicPlayer(player) })
  return player
}

function parseBody(request) {
  return new Promise((resolve, reject) => {
    let body = ""
    request.on("data", chunk => {
      body += chunk
      if (body.length > 100000) request.destroy()
    })
    request.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {})
      } catch (error) {
        reject(error)
      }
    })
    request.on("error", reject)
  })
}

function handleAction(data, response) {
  const player = playerFor(data.playerId)
  if (!player) {
    sendJson(response, 401, { error: "Player not found" })
    return
  }
  player.lastSeen = Date.now()

  if (data.action === "cursor") {
    player.x = Math.max(0, Math.min(2100, Number(data.x) || 0))
    player.y = Math.max(0, Math.min(1500, Number(data.y) || 0))
    broadcast({
      type: "cursor",
      player: publicPlayer(player)
    }, player.id)
    sendJson(response, 200, { ok: true })
    return
  }

  const piece = pieces[Number(data.pieceId)]
  if (!piece) {
    sendJson(response, 404, { error: "Piece not found" })
    return
  }

  if (data.action === "claim") {
    if (transitioning || piece.placed || piece.owner && piece.owner !== player.id) {
      sendJson(response, 409, { error: "Piece unavailable" })
      return
    }
    piece.owner = player.id
    broadcast({ type: "piece", piece })
    sendJson(response, 200, { piece })
    return
  }

  if (piece.owner !== player.id || piece.placed) {
    sendJson(response, 409, { error: "Piece unavailable" })
    return
  }

  const x = Math.max(0, Math.min(2050, Number(data.x) || 0))
  const y = Math.max(0, Math.min(1450, Number(data.y) || 0))

  if (data.action === "move") {
    piece.x = x
    piece.y = y
    broadcast({ type: "piece", piece }, player.id)
    sendJson(response, 200, { piece })
    return
  }

  if (data.action === "drop") {
    const targetX = boardX + piece.column * tile
    const targetY = boardY + piece.row * tile
    const close = Math.hypot(x - targetX, y - targetY) < 34
    piece.x = close ? targetX : x
    piece.y = close ? targetY : y
    piece.placed = close
    piece.owner = null
    if (close) player.placed += 1
    const placedCount = progress()
    broadcast({
      type: "piece",
      piece,
      player: publicPlayer(player),
      progress: placedCount
    })
    sendJson(response, 200, {
      piece,
      placed: close,
      progress: placedCount
    })
    if (placedCount === pieces.length) advanceLevel()
    return
  }

  sendJson(response, 400, { error: "Unknown action" })
}

function serveFile(request, response) {
  const pathname = new URL(request.url, "http://localhost").pathname
  const requested = pathname === "/" ? "/index.html" : pathname
  const safePath = path.normalize(requested).replace(/^(\.\.[/\\])+/, "")
  const filePath = path.join(publicDir, safePath)
  if (!filePath.startsWith(publicDir)) {
    response.writeHead(403)
    response.end()
    return
  }
  fs.readFile(filePath, (error, content) => {
    if (error) {
      response.writeHead(404)
      response.end("Not found")
      return
    }
    response.writeHead(200, {
      "content-type": types[path.extname(filePath)] || "application/octet-stream",
      "cache-control": pathname === "/" ? "no-store" : "public, max-age=3600"
    })
    response.end(content)
  })
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url, "http://localhost")

  if (request.method === "GET" && url.pathname === "/health") {
    sendJson(response, 200, {
      status: "ok",
      players: players.size,
      placed: progress(),
      total: pieces.length,
      level,
      levels: levelCount,
      loop
    })
    return
  }

  if (request.method === "POST" && url.pathname === "/api/join") {
    try {
      const data = await parseBody(request)
      const player = join(data.playerId, data.name)
      sendJson(response, 200, {
        player: publicPlayer(player),
        players: [...players.values()].map(publicPlayer),
        pieces,
        progress: progress(),
        level,
        levels: levelCount,
        loop,
        board: { columns, rows, tile, x: boardX, y: boardY }
      })
    } catch (error) {
      sendJson(response, 400, { error: "Invalid request" })
    }
    return
  }

  if (request.method === "POST" && url.pathname === "/api/action") {
    try {
      handleAction(await parseBody(request), response)
    } catch (error) {
      sendJson(response, 400, { error: "Invalid request" })
    }
    return
  }

  if (request.method === "GET" && url.pathname === "/events") {
    const playerId = url.searchParams.get("playerId")
    const player = playerFor(playerId)
    if (!player) {
      response.writeHead(401)
      response.end()
      return
    }
    response.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "connection": "keep-alive"
    })
    response.write(`data: ${JSON.stringify({
      type: "state",
      players: [...players.values()].map(publicPlayer),
      pieces,
      progress: progress(),
      level,
      levels: levelCount,
      loop
    })}\n\n`)
    const client = { id: player.id, response }
    clients.add(client)
    request.on("close", () => clients.delete(client))
    return
  }

  if (request.method === "GET") {
    serveFile(request, response)
    return
  }

  response.writeHead(405)
  response.end()
})

setInterval(() => {
  for (const client of clients) {
    const player = players.get(client.id)
    if (player) player.lastSeen = Date.now()
  }
  const cutoff = Date.now() - 30000
  for (const [id, player] of players) {
    if (player.lastSeen < cutoff) {
      players.delete(id)
      for (const piece of pieces) {
        if (piece.owner === id) piece.owner = null
      }
      broadcast({ type: "leave", playerId: id })
    }
  }
  for (const client of clients) client.response.write(": keepalive\n\n")
}, 10000)

server.listen(port, "0.0.0.0", () => {
  process.stdout.write(`Big Cat Jigsaw listening on http://localhost:${port}\n`)
})
