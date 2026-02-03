import React, { useEffect, useMemo, useRef, useState } from "https://esm.sh/react@19.0.0"
import { createRoot } from "https://esm.sh/react-dom@19.0.0/client"

const themes = {
  Classic: {
    bg: "#0f0f10",
    panel: "#16181d",
    text: "#e7e7e7",
    grid: "#1e2129",
    cellBorder: "#2b2f3a",
  },
  Neon: {
    bg: "#05030a",
    panel: "#0f0b1f",
    text: "#e6f6ff",
    grid: "#120f24",
    cellBorder: "#2b1f55",
  },
  Light: {
    bg: "#f7f7f9",
    panel: "#ffffff",
    text: "#121217",
    grid: "#e7e7ee",
    cellBorder: "#d5d5de",
  },
}

const tetrominoes = [
  { name: "I", color: "#00f0f0", shape: [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]] },
  { name: "J", color: "#0000f0", shape: [[1,0,0],[1,1,1],[0,0,0]] },
  { name: "L", color: "#f0a000", shape: [[0,0,1],[1,1,1],[0,0,0]] },
  { name: "O", color: "#f0f000", shape: [[1,1],[1,1]] },
  { name: "S", color: "#00f000", shape: [[0,1,1],[1,1,0],[0,0,0]] },
  { name: "T", color: "#a000f0", shape: [[0,1,0],[1,1,1],[0,0,0]] },
  { name: "Z", color: "#f00000", shape: [[1,1,0],[0,1,1],[0,0,0]] },
]

const defaultSettings = {
  theme: "Classic",
  difficulty: "Normal",
  width: 10,
  height: 20,
  pointsPerLine: 10,
  pointsPerLevel: 50,
  baseDropMs: 500,
  minDropMs: 120,
  freezeIntervalMs: 30000,
  freezeDurationMs: 10000,
  freezeChance: 0.5,
  growIntervalMs: 50000,
  growStep: 0.06,
  showGhost: true,
  showNext: true,
}

const difficultyOptions = {
  Easy: 700,
  Normal: 500,
  Hard: 350,
}

const createEmptyGrid = (width, height) => {
  const rows = []
  for (let y = 0; y < height; y += 1) {
    const row = []
    for (let x = 0; x < width; x += 1) {
      row.push(null)
    }
    rows.push(row)
  }
  return rows
}

const rotateMatrix = (matrix) => {
  const size = matrix.length
  const rotated = []
  for (let y = 0; y < size; y += 1) {
    const row = []
    for (let x = 0; x < size; x += 1) {
      row.push(matrix[size - x - 1][y])
    }
    rotated.push(row)
  }
  return rotated
}

const randomPiece = () => {
  const base = tetrominoes[Math.floor(Math.random() * tetrominoes.length)]
  return { name: base.name, color: base.color, shape: base.shape.map((row) => row.slice()) }
}

const canPlace = (grid, piece, pos) => {
  for (let y = 0; y < piece.shape.length; y += 1) {
    for (let x = 0; x < piece.shape[y].length; x += 1) {
      if (piece.shape[y][x]) {
        const gx = pos.x + x
        const gy = pos.y + y
        if (gx < 0 || gx >= grid[0].length || gy >= grid.length) {
          return false
        }
        if (gy >= 0 && grid[gy][gx]) {
          return false
        }
      }
    }
  }
  return true
}

const mergePiece = (grid, piece, pos) => {
  const merged = grid.map((row) => row.slice())
  for (let y = 0; y < piece.shape.length; y += 1) {
    for (let x = 0; x < piece.shape[y].length; x += 1) {
      if (piece.shape[y][x]) {
        const gx = pos.x + x
        const gy = pos.y + y
        if (gy >= 0 && gy < merged.length && gx >= 0 && gx < merged[0].length) {
          merged[gy][gx] = piece.color
        }
      }
    }
  }
  return merged
}

const clearLines = (grid) => {
  const remaining = grid.filter((row) => row.some((cell) => !cell))
  const cleared = grid.length - remaining.length
  const newRows = []
  for (let i = 0; i < cleared; i += 1) {
    newRows.push(new Array(grid[0].length).fill(null))
  }
  return { grid: [...newRows, ...remaining], cleared }
}

const computeDropMs = (settings, level) => {
  const base = settings.baseDropMs
  const reduction = (level - 1) * 30
  return Math.max(settings.minDropMs, base - reduction)
}

const getGhostPosition = (grid, piece, pos) => {
  let ghostY = pos.y
  while (canPlace(grid, piece, { x: pos.x, y: ghostY + 1 })) {
    ghostY += 1
  }
  return { x: pos.x, y: ghostY }
}

const App = () => {
  const [settings, setSettings] = useState(defaultSettings)
  const [grid, setGrid] = useState(() => createEmptyGrid(defaultSettings.width, defaultSettings.height))
  const [current, setCurrent] = useState(randomPiece)
  const [next, setNext] = useState(randomPiece)
  const [position, setPosition] = useState({ x: 3, y: -1 })
  const [score, setScore] = useState(0)
  const [level, setLevel] = useState(1)
  const [running, setRunning] = useState(false)
  const [gameOver, setGameOver] = useState(false)
  const [freeze, setFreeze] = useState(false)
  const [scale, setScale] = useState(1)
  const [showAdmin, setShowAdmin] = useState(false)
  const freezeTimeoutRef = useRef(null)

  const theme = themes[settings.theme] || themes.Classic

  const resetGame = (opts = settings) => {
    const startX = Math.floor(opts.width / 2) - 2
    setGrid(createEmptyGrid(opts.width, opts.height))
    setCurrent(randomPiece())
    setNext(randomPiece())
    setPosition({ x: startX, y: -1 })
    setScore(0)
    setLevel(1)
    setGameOver(false)
    setFreeze(false)
    setScale(1)
    setRunning(true)
  }

  const spawnPiece = (baseGrid, baseNext) => {
    const startX = Math.floor(settings.width / 2) - 2
    const nextPiece = baseNext || randomPiece()
    const newPos = { x: startX, y: -1 }
    if (!canPlace(baseGrid, nextPiece, newPos)) {
      setGameOver(true)
      setRunning(false)
      return
    }
    setCurrent(nextPiece)
    setNext(randomPiece())
    setPosition(newPos)
  }

  const lockPiece = () => {
    const merged = mergePiece(grid, current, position)
    const cleared = clearLines(merged)
    const added = cleared.cleared * settings.pointsPerLine
    const newScore = score + added
    const newLevel = Math.floor(newScore / settings.pointsPerLevel) + 1
    setGrid(cleared.grid)
    setScore(newScore)
    setLevel(newLevel)
    spawnPiece(cleared.grid, next)
  }

  const move = (dx, dy) => {
    const nextPos = { x: position.x + dx, y: position.y + dy }
    if (canPlace(grid, current, nextPos)) {
      setPosition(nextPos)
      return true
    }
    if (dy === 1) {
      lockPiece()
    }
    return false
  }

  const hardDrop = () => {
    let nextPos = { ...position }
    while (canPlace(grid, current, { x: nextPos.x, y: nextPos.y + 1 })) {
      nextPos = { x: nextPos.x, y: nextPos.y + 1 }
    }
    setPosition(nextPos)
    setTimeout(() => lockPiece(), 0)
  }

  const rotate = () => {
    const rotated = rotateMatrix(current.shape)
    const rotatedPiece = { ...current, shape: rotated }
    const kicks = [0, -1, 1, -2, 2]
    for (const dx of kicks) {
      const nextPos = { x: position.x + dx, y: position.y }
      if (canPlace(grid, rotatedPiece, nextPos)) {
        setCurrent(rotatedPiece)
        setPosition(nextPos)
        return
      }
    }
  }

  useEffect(() => {
    if (!running || freeze || gameOver) {
      return undefined
    }
    const dropMs = computeDropMs(settings, level)
    const timer = setInterval(() => {
      move(0, 1)
    }, dropMs)
    return () => clearInterval(timer)
  }, [running, freeze, gameOver, level, position, current, grid, settings])

  useEffect(() => {
    if (!running || gameOver) {
      return undefined
    }
    const timer = setInterval(() => {
      if (Math.random() < settings.freezeChance) {
        setFreeze(true)
        if (freezeTimeoutRef.current) {
          clearTimeout(freezeTimeoutRef.current)
        }
        freezeTimeoutRef.current = setTimeout(() => {
          setFreeze(false)
        }, settings.freezeDurationMs)
      }
    }, settings.freezeIntervalMs)
    return () => clearInterval(timer)
  }, [running, gameOver, settings])

  useEffect(() => {
    if (!running || gameOver) {
      return undefined
    }
    const timer = setInterval(() => {
      setScale((value) => Math.min(1.8, value + settings.growStep))
    }, settings.growIntervalMs)
    return () => clearInterval(timer)
  }, [running, gameOver, settings])

  useEffect(() => {
    const handler = (event) => {
      if (!running || gameOver || freeze) {
        return
      }
      if (event.key === "ArrowLeft") {
        move(-1, 0)
      } else if (event.key === "ArrowRight") {
        move(1, 0)
      } else if (event.key === "ArrowDown") {
        move(0, 1)
      } else if (event.key === "ArrowUp") {
        rotate()
      } else if (event.key === " ") {
        hardDrop()
      } else if (event.key === "p" || event.key === "P") {
        setRunning((value) => !value)
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [running, gameOver, freeze, position, current, grid])

  const ghostPosition = useMemo(() => {
    if (!settings.showGhost || freeze || gameOver) {
      return null
    }
    return getGhostPosition(grid, current, position)
  }, [grid, current, position, settings.showGhost, freeze, gameOver])

  const gridWithCurrent = useMemo(() => {
    const merged = mergePiece(grid, current, position)
    if (!ghostPosition || position.y === ghostPosition.y) {
      return merged
    }
    const ghostGrid = merged.map((row) => row.slice())
    for (let y = 0; y < current.shape.length; y += 1) {
      for (let x = 0; x < current.shape[y].length; x += 1) {
        if (current.shape[y][x]) {
          const gx = ghostPosition.x + x
          const gy = ghostPosition.y + y
          if (gy >= 0 && gy < ghostGrid.length && gx >= 0 && gx < ghostGrid[0].length) {
            if (!ghostGrid[gy][gx]) {
              ghostGrid[gy][gx] = "rgba(255,255,255,0.2)"
            }
          }
        }
      }
    }
    return ghostGrid
  }, [grid, current, position, ghostPosition])

  const applySettings = (nextSettings) => {
    let baseDrop = nextSettings.baseDropMs
    if (nextSettings.difficulty !== "Custom") {
      baseDrop = difficultyOptions[nextSettings.difficulty] || nextSettings.baseDropMs
    }
    const mergedSettings = { ...nextSettings, baseDropMs: baseDrop }
    setSettings(mergedSettings)
    resetGame(mergedSettings)
  }

  const updateSettings = (field, value) => {
    setSettings((prev) => ({ ...prev, [field]: value }))
  }

  const cellSize = Math.max(20, Math.floor(360 / settings.width))

  return (
    React.createElement(
      "div",
      { className: "app", style: { "--bg": theme.bg, "--panel": theme.panel, "--text": theme.text, "--grid": theme.grid, "--cellBorder": theme.cellBorder } },
      React.createElement("header", { className: "header" },
        React.createElement("h1", null, "Tetris"),
        React.createElement("div", { className: "header-actions" },
          React.createElement("button", { onClick: () => setShowAdmin((value) => !value) }, showAdmin ? "Hide Admin" : "Show Admin"),
          React.createElement("button", { onClick: () => resetGame() }, "Start"),
          React.createElement("button", { onClick: () => setRunning((value) => !value) }, running ? "Pause" : "Resume")
        )
      ),
      React.createElement("main", { className: "main" },
        React.createElement("section", { className: "board-wrapper" },
          React.createElement("div", { className: "board", style: { transform: `scale(${scale})`, gridTemplateColumns: `repeat(${settings.width}, ${cellSize}px)` } },
            gridWithCurrent.map((row, y) => row.map((cell, x) =>
              React.createElement("div", {
                key: `${y}-${x}`,
                className: "cell",
                style: { width: `${cellSize}px`, height: `${cellSize}px`, background: cell || "transparent" }
              })
            ))
          ),
          freeze && React.createElement("div", { className: "freeze" }, "Freeze")
        ),
        React.createElement("section", { className: "panel" },
          React.createElement("div", { className: "stat" }, React.createElement("span", null, "Score"), React.createElement("strong", null, score)),
          React.createElement("div", { className: "stat" }, React.createElement("span", null, "Level"), React.createElement("strong", null, level)),
          React.createElement("div", { className: "stat" }, React.createElement("span", null, "Drop"), React.createElement("strong", null, `${computeDropMs(settings, level)} ms`)),
          React.createElement("div", { className: "stat" }, React.createElement("span", null, "Status"), React.createElement("strong", null, gameOver ? "Game Over" : running ? freeze ? "Frozen" : "Running" : "Stopped")),
          settings.showNext && React.createElement("div", { className: "next" },
            React.createElement("span", null, "Next"),
            React.createElement("div", { className: "next-grid" },
              next.shape.map((row, y) => row.map((cell, x) =>
                React.createElement("div", { key: `${y}-${x}`, className: "next-cell", style: { background: cell ? next.color : "transparent" } })
              ))
            )
          ),
          React.createElement("div", { className: "controls" },
            React.createElement("div", null, "Move: Arrow keys"),
            React.createElement("div", null, "Rotate: Arrow up"),
            React.createElement("div", null, "Drop: Space"),
            React.createElement("div", null, "Pause: P")
          )
        )
      ),
      showAdmin && React.createElement("section", { className: "admin" },
        React.createElement("h2", null, "Admin"),
        React.createElement("div", { className: "admin-grid" },
          React.createElement("label", null, "Theme", React.createElement("select", { value: settings.theme, onChange: (e) => updateSettings("theme", e.target.value) }, Object.keys(themes).map((name) => React.createElement("option", { key: name, value: name }, name)))),
          React.createElement("label", null, "Difficulty", React.createElement("select", { value: settings.difficulty, onChange: (e) => updateSettings("difficulty", e.target.value) }, ["Easy","Normal","Hard","Custom"].map((name) => React.createElement("option", { key: name, value: name }, name)))),
          React.createElement("label", null, "Base Drop (ms)", React.createElement("input", { type: "number", min: 100, value: settings.baseDropMs, onChange: (e) => updateSettings("baseDropMs", Number(e.target.value)) })),
          React.createElement("label", null, "Min Drop (ms)", React.createElement("input", { type: "number", min: 60, value: settings.minDropMs, onChange: (e) => updateSettings("minDropMs", Number(e.target.value)) })),
          React.createElement("label", null, "Board Width", React.createElement("input", { type: "number", min: 6, max: 16, value: settings.width, onChange: (e) => updateSettings("width", Number(e.target.value)) })),
          React.createElement("label", null, "Board Height", React.createElement("input", { type: "number", min: 12, max: 28, value: settings.height, onChange: (e) => updateSettings("height", Number(e.target.value)) })),
          React.createElement("label", null, "Points Per Line", React.createElement("input", { type: "number", min: 1, value: settings.pointsPerLine, onChange: (e) => updateSettings("pointsPerLine", Number(e.target.value)) })),
          React.createElement("label", null, "Points Per Level", React.createElement("input", { type: "number", min: 10, value: settings.pointsPerLevel, onChange: (e) => updateSettings("pointsPerLevel", Number(e.target.value)) })),
          React.createElement("label", null, "Freeze Interval (ms)", React.createElement("input", { type: "number", min: 5000, value: settings.freezeIntervalMs, onChange: (e) => updateSettings("freezeIntervalMs", Number(e.target.value)) })),
          React.createElement("label", null, "Freeze Duration (ms)", React.createElement("input", { type: "number", min: 1000, value: settings.freezeDurationMs, onChange: (e) => updateSettings("freezeDurationMs", Number(e.target.value)) })),
          React.createElement("label", null, "Freeze Chance", React.createElement("input", { type: "number", min: 0, max: 1, step: 0.1, value: settings.freezeChance, onChange: (e) => updateSettings("freezeChance", Number(e.target.value)) })),
          React.createElement("label", null, "Grow Interval (ms)", React.createElement("input", { type: "number", min: 5000, value: settings.growIntervalMs, onChange: (e) => updateSettings("growIntervalMs", Number(e.target.value)) })),
          React.createElement("label", null, "Grow Step", React.createElement("input", { type: "number", min: 0.01, max: 0.2, step: 0.01, value: settings.growStep, onChange: (e) => updateSettings("growStep", Number(e.target.value)) })),
          React.createElement("label", { className: "toggle" }, React.createElement("input", { type: "checkbox", checked: settings.showGhost, onChange: (e) => updateSettings("showGhost", e.target.checked) }), "Ghost Piece"),
          React.createElement("label", { className: "toggle" }, React.createElement("input", { type: "checkbox", checked: settings.showNext, onChange: (e) => updateSettings("showNext", e.target.checked) }), "Show Next")
        ),
        React.createElement("div", { className: "admin-actions" },
          React.createElement("button", { onClick: () => applySettings(settings) }, "Apply")
        )
      )
    )
  )
}

const root = createRoot(document.getElementById("root"))
root.render(React.createElement(App))
