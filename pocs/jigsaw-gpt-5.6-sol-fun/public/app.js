const stage = document.querySelector("#stage")
const world = document.querySelector("#world")
const loosePieces = document.querySelector("#loosePieces")
const cursors = document.querySelector("#cursors")
const tray = document.querySelector("#tray")
const dragLayer = document.querySelector("#dragLayer")
const progressText = document.querySelector("#progressText")
const progressBar = document.querySelector("#progressBar")
const percentText = document.querySelector("#percentText")
const looseCount = document.querySelector("#looseCount")
const crewList = document.querySelector("#crewList")
const onlineCount = document.querySelector("#onlineCount")
const identityDot = document.querySelector("#identityDot")
const playerName = document.querySelector("#playerName")
const playerScore = document.querySelector("#playerScore")
const zoomOutput = document.querySelector("#zoomOutput")
const toast = document.querySelector("#toast")
const levelText = document.querySelector("#levelText")
const artTitle = document.querySelector("#artTitle")
const depthLabel = document.querySelector("#depthLabel")
const depthMap = document.querySelector("#depthMap")
const referenceImage = document.querySelector("#referenceImage")
const nextWorld = document.querySelector("#nextWorld")
const portal = document.querySelector("#portal")
const portalTitle = document.querySelector("#portalTitle")
const portalLevel = document.querySelector("#portalLevel")
const playerId = localStorage.getItem("big-cat-player") || crypto.randomUUID()
localStorage.setItem("big-cat-player", playerId)

const state = {
  player: null,
  players: new Map(),
  pieces: new Map(),
  board: { columns: 21, rows: 15, tile: 50, x: 525, y: 360 },
  zoom: .8,
  dragging: null,
  pan: null,
  progress: 0,
  level: 1,
  loop: 1,
  lastCursor: 0
}

const levels = [
  {
    title: "The Great Cat Nap",
    card: "A very crowded cat nap",
    image: "/puzzle.jpg",
    alt: "Many cats gathered around a cozy bed"
  },
  {
    title: "The Endless Library",
    card: "Books all the way down",
    image: "/levels/level-2.png",
    alt: "Cats reading in a towering old library"
  },
  {
    title: "The Ocean Between Pages",
    card: "Above and below the waves",
    image: "/levels/level-3.png",
    alt: "Whales, boats, fish and coral in a bright ocean"
  },
  {
    title: "Kingdom Made of Sand",
    card: "A castle the tide forgot",
    image: "/levels/level-4.png",
    alt: "A sprawling sandcastle on a sunny beach"
  },
  {
    title: "The Daylight Carnival",
    card: "Fun beneath the kites",
    image: "/levels/level-5.png",
    alt: "A seaside carnival with a ferris wheel and kites"
  },
  {
    title: "The Rainy City",
    card: "Puddles open secret doors",
    image: "/levels/level-6.png",
    alt: "Cats and frogs walking through a glowing city in the rain"
  },
  {
    title: "The Garden After Rain",
    card: "Everything grows inward",
    image: "/levels/level-7.png",
    alt: "A lush secret garden filled with flowers and birds"
  },
  {
    title: "The Observatory Beyond",
    card: "The sky becomes a room",
    image: "/levels/level-8.png",
    alt: "Cats in a floating observatory among stars and planets"
  },
  {
    title: "The Puzzle That Builds You",
    card: "Back where the pieces began",
    image: "/levels/level-9.png",
    alt: "Cats assembling a recursive celestial jigsaw"
  }
]

function request(path, body) {
  return fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body)
  }).then(response => {
    if (!response.ok) throw new Error("Request failed")
    return response.json()
  })
}

function pieceStyle(piece) {
  return `background-position: -${piece.column * 50}px -${piece.row * 50}px`
}

function renderLevel(level, loop) {
  state.level = level
  state.loop = loop
  const current = levels[level - 1]
  const following = levels[level % levels.length]
  document.documentElement.style.setProperty("--art-url", `url("${current.image}")`)
  levelText.textContent = `LEVEL ${level} OF 9 · ${current.title.toUpperCase()}`
  artTitle.textContent = current.card
  depthLabel.textContent = `DEPTH ${level} · LOOP ${loop}`
  referenceImage.src = current.image
  referenceImage.alt = current.alt
  nextWorld.textContent = `Next world: ${following.title}`
  depthMap.replaceChildren()
  for (let index = 1; index <= 9; index += 1) {
    const marker = document.createElement("span")
    marker.classList.toggle("visited", index < level)
    marker.classList.toggle("current", index === level)
    depthMap.append(marker)
  }
}

function pieceButton(piece) {
  const button = document.createElement("button")
  button.type = "button"
  button.className = `piece${piece.placed ? " placed" : ""}${piece.owner ? " held" : ""}`
  button.dataset.pieceId = piece.id
  button.setAttribute("aria-label", `Puzzle piece ${piece.id + 1}`)
  button.style.cssText = pieceStyle(piece)
  button.style.setProperty("--tilt", `${piece.id % 2 ? 3 : -3}deg`)
  button.addEventListener("pointerdown", startPieceDrag)
  return button
}

function renderPiece(piece) {
  state.pieces.set(piece.id, piece)
  let element = document.querySelector(`.piece[data-piece-id="${piece.id}"]`)
  const visibleInTray = !piece.placed && piece.x === null
  const visibleInWorld = piece.placed || piece.x !== null

  if (visibleInTray && (!element || element.parentElement !== tray)) {
    if (element) element.remove()
    element = pieceButton(piece)
    tray.append(element)
  }

  if (visibleInWorld && (!element || element.parentElement !== loosePieces)) {
    if (element) element.remove()
    element = pieceButton(piece)
    loosePieces.append(element)
  }

  if (!element) return
  element.classList.toggle("placed", piece.placed)
  element.classList.toggle("held", Boolean(piece.owner && piece.owner !== playerId))
  element.disabled = piece.placed || Boolean(piece.owner && piece.owner !== playerId)
  element.style.backgroundPosition = `-${piece.column * 50}px -${piece.row * 50}px`
  if (visibleInWorld) {
    element.style.left = `${piece.x}px`
    element.style.top = `${piece.y}px`
  } else {
    element.style.removeProperty("left")
    element.style.removeProperty("top")
  }
}

function renderAllPieces() {
  loosePieces.replaceChildren()
  tray.replaceChildren()
  for (const piece of state.pieces.values()) renderPiece(piece)
}

function cursorMarkup(player) {
  const cursor = document.createElement("div")
  cursor.className = "cursor"
  cursor.dataset.playerId = player.id
  cursor.style.setProperty("--player-color", player.color)
  cursor.innerHTML = `<svg viewBox="0 0 24 30" aria-hidden="true"><path fill="${player.color}" stroke="white" stroke-width="2" d="M2 2v22l6-6 4 10 5-2-4-9h8Z"/></svg><span>${player.name}</span>`
  return cursor
}

function renderCursor(player) {
  if (player.id === playerId) return
  let cursor = cursors.querySelector(`[data-player-id="${player.id}"]`)
  if (!cursor) {
    cursor = cursorMarkup(player)
    cursors.append(cursor)
  }
  cursor.style.left = `${player.x}px`
  cursor.style.top = `${player.y}px`
}

function renderPlayers() {
  crewList.replaceChildren()
  cursors.replaceChildren()
  const sorted = [...state.players.values()].sort((a, b) => b.placed - a.placed)
  for (const player of sorted) {
    const member = document.createElement("div")
    member.className = "crew-member"
    member.style.setProperty("--member-color", player.color)
    member.innerHTML = `<span class="crew-avatar">${player.name.charAt(0)}</span><div><strong>${player.name}${player.id === playerId ? " · you" : ""}</strong><small>${player.id === playerId ? "at your screen" : "moving around the board"}</small></div><span class="crew-score">${player.placed}</span>`
    crewList.append(member)
    renderCursor(player)
  }
  onlineCount.textContent = `${sorted.length} online`
  if (state.player) {
    const current = state.players.get(playerId) || state.player
    playerName.textContent = current.name
    playerScore.textContent = `${current.placed} ${current.placed === 1 ? "piece" : "pieces"} placed`
    identityDot.style.background = current.color
  }
}

function renderProgress(value) {
  state.progress = value
  const total = state.pieces.size || 315
  const percent = Math.round(value / total * 100)
  progressText.textContent = `${value} / ${total}`
  progressBar.style.width = `${percent}%`
  percentText.textContent = `${percent}%`
  looseCount.textContent = `${total - value} waiting`
  document.querySelector(".board-label").style.display = value > total * .12 ? "none" : ""
}

function worldPoint(event) {
  const rect = stage.getBoundingClientRect()
  return {
    x: (event.clientX - rect.left + stage.scrollLeft) / state.zoom,
    y: (event.clientY - rect.top + stage.scrollTop) / state.zoom
  }
}

async function startPieceDrag(event) {
  event.preventDefault()
  event.stopPropagation()
  const pieceId = Number(event.currentTarget.dataset.pieceId)
  const piece = state.pieces.get(pieceId)
  if (!piece || piece.placed || piece.owner && piece.owner !== playerId) return

  try {
    const result = await request("/api/action", {
      playerId,
      action: "claim",
      pieceId
    })
    Object.assign(piece, result.piece)
    const ghost = pieceButton(piece)
    ghost.removeEventListener("pointerdown", startPieceDrag)
    dragLayer.replaceChildren(ghost)
    state.dragging = {
      piece,
      ghost,
      offsetX: 25,
      offsetY: 25,
      lastMove: 0
    }
    event.currentTarget.classList.add("held")
    positionGhost(event)
    document.addEventListener("pointermove", movePieceDrag)
    document.addEventListener("pointerup", endPieceDrag, { once: true })
  } catch (error) {
    showToast("Another player has that piece")
  }
}

function positionGhost(event) {
  if (!state.dragging) return
  state.dragging.ghost.style.left = `${event.clientX - state.dragging.offsetX}px`
  state.dragging.ghost.style.top = `${event.clientY - state.dragging.offsetY}px`
}

function movePieceDrag(event) {
  if (!state.dragging) return
  positionGhost(event)
  const now = performance.now()
  if (now - state.dragging.lastMove < 70) return
  state.dragging.lastMove = now
  const point = worldPoint(event)
  request("/api/action", {
    playerId,
    action: "move",
    pieceId: state.dragging.piece.id,
    x: point.x - 25,
    y: point.y - 25
  }).catch(() => {})
}

async function endPieceDrag(event) {
  document.removeEventListener("pointermove", movePieceDrag)
  if (!state.dragging) return
  const drag = state.dragging
  state.dragging = null
  const point = worldPoint(event)
  try {
    const result = await request("/api/action", {
      playerId,
      action: "drop",
      pieceId: drag.piece.id,
      x: point.x - 25,
      y: point.y - 25
    })
    renderPiece(result.piece)
    renderProgress(result.progress)
    if (result.placed) {
      showToast("Perfect fit")
      const current = state.players.get(playerId)
      if (current) {
        current.placed += 1
        renderPlayers()
      }
    }
  } catch (error) {
    showToast("The piece slipped away")
  }
  dragLayer.replaceChildren()
}

function showToast(message) {
  toast.textContent = message
  toast.classList.add("show")
  clearTimeout(showToast.timer)
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 1800)
}

function setZoom(next) {
  state.zoom = Math.max(.45, Math.min(1.2, next))
  document.documentElement.style.setProperty("--world-scale", state.zoom)
  zoomOutput.textContent = `${Math.round(state.zoom * 100)}%`
}

function recenter() {
  const boardCenterX = (state.board.x + state.board.columns * state.board.tile / 2) * state.zoom
  const boardCenterY = (state.board.y + state.board.rows * state.board.tile / 2) * state.zoom
  stage.scrollTo({
    left: boardCenterX - stage.clientWidth / 2,
    top: boardCenterY - stage.clientHeight / 2,
    behavior: "smooth"
  })
}

function connectEvents() {
  const events = new EventSource(`/events?playerId=${encodeURIComponent(playerId)}`)
  events.onmessage = event => {
    const message = JSON.parse(event.data)
    if (message.type === "state") {
      state.players = new Map(message.players.map(player => [player.id, player]))
      state.pieces = new Map(message.pieces.map(piece => [piece.id, piece]))
      renderAllPieces()
      renderPlayers()
      renderProgress(message.progress)
      renderLevel(message.level, message.loop)
    }
    if (message.type === "player") {
      state.players.set(message.player.id, message.player)
      renderPlayers()
    }
    if (message.type === "cursor") {
      state.players.set(message.player.id, message.player)
      renderCursor(message.player)
    }
    if (message.type === "piece") {
      renderPiece(message.piece)
      if (message.player) state.players.set(message.player.id, message.player)
      if (typeof message.progress === "number") renderProgress(message.progress)
      if (message.player) renderPlayers()
    }
    if (message.type === "leave") {
      state.players.delete(message.playerId)
      renderPlayers()
    }
    if (message.type === "complete") {
      const destination = levels[message.nextLevel - 1]
      portalTitle.textContent = destination.title
      portalLevel.textContent = `Opening level ${message.nextLevel} · loop ${message.nextLoop}`
      portal.classList.add("open")
    }
    if (message.type === "level") {
      state.pieces = new Map(message.pieces.map(piece => [piece.id, piece]))
      renderLevel(message.level, message.loop)
      renderAllPieces()
      renderProgress(message.progress)
      recenter()
      setTimeout(() => portal.classList.remove("open"), 550)
    }
  }
}

stage.addEventListener("pointermove", event => {
  if (!state.player || state.dragging || state.pan) return
  const now = performance.now()
  if (now - state.lastCursor < 90) return
  state.lastCursor = now
  const point = worldPoint(event)
  request("/api/action", {
    playerId,
    action: "cursor",
    x: point.x,
    y: point.y
  }).catch(() => {})
})

stage.addEventListener("pointerdown", event => {
  if (event.target.closest(".piece") || event.button !== 0) return
  state.pan = {
    x: event.clientX,
    y: event.clientY,
    left: stage.scrollLeft,
    top: stage.scrollTop
  }
  stage.classList.add("dragging-space")
})

stage.addEventListener("pointermove", event => {
  if (!state.pan) return
  stage.scrollLeft = state.pan.left - (event.clientX - state.pan.x)
  stage.scrollTop = state.pan.top - (event.clientY - state.pan.y)
})

document.addEventListener("pointerup", () => {
  state.pan = null
  stage.classList.remove("dragging-space")
})

document.querySelector("#recenterButton").addEventListener("click", recenter)
document.querySelector("#zoomInButton").addEventListener("click", () => setZoom(state.zoom + .1))
document.querySelector("#zoomOutButton").addEventListener("click", () => setZoom(state.zoom - .1))
document.querySelector("#previewButton").addEventListener("pointerdown", () => document.body.classList.add("peek"))
document.querySelector("#previewButton").addEventListener("pointerup", () => document.body.classList.remove("peek"))
document.querySelector("#previewButton").addEventListener("pointerleave", () => document.body.classList.remove("peek"))
document.querySelector("#copyButton").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(location.href)
    showToast("Room link copied")
  } catch (error) {
    showToast("Copy the address from your browser")
  }
})

request("/api/join", { playerId })
  .then(data => {
    state.player = data.player
    state.board = data.board
    state.players = new Map(data.players.map(player => [player.id, player]))
    state.pieces = new Map(data.pieces.map(piece => [piece.id, piece]))
    renderAllPieces()
    renderPlayers()
    renderProgress(data.progress)
    renderLevel(data.level, data.loop)
    connectEvents()
    requestAnimationFrame(recenter)
  })
  .catch(() => {
    playerName.textContent = "Room unavailable"
    showToast("Could not join the room")
  })
