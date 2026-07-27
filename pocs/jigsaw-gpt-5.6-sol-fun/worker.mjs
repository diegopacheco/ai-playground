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

const encoder = new TextEncoder()

function json(data, status = 200) {
  return Response.json(data, {
    status,
    headers: { "cache-control": "no-store" }
  })
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

function broadcast(data, excludedId) {
  const payload = encoder.encode(`data: ${JSON.stringify(data)}\n\n`)
  for (const client of clients) {
    if (client.id === excludedId) continue
    try {
      client.controller.enqueue(payload)
    } catch (error) {
      clients.delete(client)
    }
  }
}

function join(id, requestedName) {
  const cleanId = String(id || crypto.randomUUID()).slice(0, 80)
  const existing = players.get(cleanId)
  if (existing) return existing
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
    placed: 0
  }
  players.set(cleanId, player)
  broadcast({ type: "player", player: publicPlayer(player) })
  return player
}

function advanceLevel(context) {
  if (transitioning) return
  transitioning = true
  const nextLevel = level === levelCount ? 1 : level + 1
  const nextLoop = nextLevel === 1 ? loop + 1 : loop
  broadcast({ type: "complete", level, loop, nextLevel, nextLoop })
  context.waitUntil(new Promise(resolve => {
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
      resolve()
    }, 1800)
  }))
}

function handleAction(data, context) {
  const player = players.get(String(data.playerId || ""))
  if (!player) return json({ error: "Player not found" }, 401)

  if (data.action === "cursor") {
    player.x = Math.max(0, Math.min(2100, Number(data.x) || 0))
    player.y = Math.max(0, Math.min(1500, Number(data.y) || 0))
    broadcast({ type: "cursor", player: publicPlayer(player) }, player.id)
    return json({ ok: true })
  }

  const piece = pieces[Number(data.pieceId)]
  if (!piece) return json({ error: "Piece not found" }, 404)

  if (data.action === "claim") {
    if (transitioning || piece.placed || piece.owner && piece.owner !== player.id) {
      return json({ error: "Piece unavailable" }, 409)
    }
    piece.owner = player.id
    broadcast({ type: "piece", piece })
    return json({ piece })
  }

  if (piece.owner !== player.id || piece.placed) {
    return json({ error: "Piece unavailable" }, 409)
  }

  const x = Math.max(0, Math.min(2050, Number(data.x) || 0))
  const y = Math.max(0, Math.min(1450, Number(data.y) || 0))

  if (data.action === "move") {
    piece.x = x
    piece.y = y
    broadcast({ type: "piece", piece }, player.id)
    return json({ piece })
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
    if (placedCount === pieces.length) advanceLevel(context)
    return json({ piece, placed: close, progress: placedCount })
  }

  return json({ error: "Unknown action" }, 400)
}

function events(request, player) {
  let client
  const stream = new ReadableStream({
    start(controller) {
      client = { id: player.id, controller }
      clients.add(client)
      controller.enqueue(encoder.encode(`data: ${JSON.stringify({
        type: "state",
        players: [...players.values()].map(publicPlayer),
        pieces,
        progress: progress(),
        level,
        levels: levelCount,
        loop
      })}\n\n`))
    },
    cancel() {
      clients.delete(client)
    }
  })
  request.signal.addEventListener("abort", () => clients.delete(client))
  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "connection": "keep-alive"
    }
  })
}

export default {
  async fetch(request, environment, context) {
    const url = new URL(request.url)

    if (request.method === "GET" && url.pathname === "/health") {
      return json({
        status: "ok",
        players: players.size,
        placed: progress(),
        total: pieces.length,
        level,
        levels: levelCount,
        loop
      })
    }

    if (request.method === "POST" && url.pathname === "/api/join") {
      try {
        const data = await request.json()
        const player = join(data.playerId, data.name)
        return json({
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
        return json({ error: "Invalid request" }, 400)
      }
    }

    if (request.method === "POST" && url.pathname === "/api/action") {
      try {
        return handleAction(await request.json(), context)
      } catch (error) {
        return json({ error: "Invalid request" }, 400)
      }
    }

    if (request.method === "GET" && url.pathname === "/events") {
      const player = players.get(String(url.searchParams.get("playerId") || ""))
      if (!player) return json({ error: "Player not found" }, 401)
      return events(request, player)
    }

    return environment.ASSETS.fetch(request)
  }
}
