import { useState, useCallback, useEffect, useRef } from 'react'

const TETROMINOS = {
  I: { shape: [[1, 1, 1, 1]], color: '#00f0f0' },
  O: { shape: [[1, 1], [1, 1]], color: '#f0f000' },
  T: { shape: [[0, 1, 0], [1, 1, 1]], color: '#a000f0' },
  S: { shape: [[0, 1, 1], [1, 1, 0]], color: '#00f000' },
  Z: { shape: [[1, 1, 0], [0, 1, 1]], color: '#f00000' },
  J: { shape: [[1, 0, 0], [1, 1, 1]], color: '#0000f0' },
  L: { shape: [[0, 0, 1], [1, 1, 1]], color: '#f0a000' }
}

const TETROMINO_KEYS = Object.keys(TETROMINOS)

const createEmptyBoard = (width, height) => {
  return Array.from({ length: height }, () => Array(width).fill(null))
}

const randomTetromino = () => {
  const key = TETROMINO_KEYS[Math.floor(Math.random() * TETROMINO_KEYS.length)]
  return { ...TETROMINOS[key], type: key }
}

const rotate = (matrix) => {
  const rows = matrix.length
  const cols = matrix[0].length
  const rotated = Array.from({ length: cols }, () => Array(rows).fill(0))
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      rotated[c][rows - 1 - r] = matrix[r][c]
    }
  }
  return rotated
}

export const useTetris = (initialWidth = 10, initialHeight = 20, growInterval = 30000) => {
  const [boardWidth, setBoardWidth] = useState(initialWidth)
  const [boardHeight, setBoardHeight] = useState(initialHeight)
  const [board, setBoard] = useState(() => createEmptyBoard(initialWidth, initialHeight))
  const [currentPiece, setCurrentPiece] = useState(null)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [score, setScore] = useState(0)
  const [moves, setMoves] = useState(0)
  const [level, setLevel] = useState(1)
  const [gameOver, setGameOver] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [elapsedTime, setElapsedTime] = useState(0)
  const [isFrozen, setIsFrozen] = useState(false)
  const [freezeTimeLeft, setFreezeTimeLeft] = useState(0)

  const gameLoopRef = useRef(null)
  const timeRef = useRef(null)
  const growRef = useRef(null)
  const freezeRef = useRef(null)
  const freezeCountdownRef = useRef(null)

  const checkCollision = useCallback((piece, pos, currentBoard) => {
    if (!piece) return false
    for (let y = 0; y < piece.shape.length; y++) {
      for (let x = 0; x < piece.shape[y].length; x++) {
        if (piece.shape[y][x]) {
          const newX = pos.x + x
          const newY = pos.y + y
          if (newX < 0 || newX >= currentBoard[0].length || newY >= currentBoard.length) {
            return true
          }
          if (newY >= 0 && currentBoard[newY][newX]) {
            return true
          }
        }
      }
    }
    return false
  }, [])

  const mergePieceToBoard = useCallback((piece, pos, currentBoard) => {
    const newBoard = currentBoard.map(row => [...row])
    for (let y = 0; y < piece.shape.length; y++) {
      for (let x = 0; x < piece.shape[y].length; x++) {
        if (piece.shape[y][x]) {
          const boardY = pos.y + y
          const boardX = pos.x + x
          if (boardY >= 0 && boardY < newBoard.length && boardX >= 0 && boardX < newBoard[0].length) {
            newBoard[boardY][boardX] = piece.color
          }
        }
      }
    }
    return newBoard
  }, [])

  const clearLines = useCallback((currentBoard) => {
    let linesCleared = 0
    const newBoard = currentBoard.filter(row => {
      const isFull = row.every(cell => cell !== null)
      if (isFull) linesCleared++
      return !isFull
    })
    while (newBoard.length < currentBoard.length) {
      newBoard.unshift(Array(currentBoard[0].length).fill(null))
    }
    return { newBoard, linesCleared }
  }, [])

  const spawnPiece = useCallback(() => {
    const piece = randomTetromino()
    const startX = Math.floor((board[0].length - piece.shape[0].length) / 2)
    const startPos = { x: startX, y: 0 }

    if (checkCollision(piece, startPos, board)) {
      setGameOver(true)
      return
    }

    setCurrentPiece(piece)
    setPosition(startPos)
  }, [board, checkCollision])

  const lockPiece = useCallback(() => {
    if (!currentPiece) return

    const mergedBoard = mergePieceToBoard(currentPiece, position, board)
    const { newBoard, linesCleared } = clearLines(mergedBoard)

    if (linesCleared > 0) {
      const points = linesCleared * 10
      setScore(prev => {
        const newScore = prev + points
        const newLevel = Math.floor(newScore / 100) + 1
        setLevel(newLevel)
        return newScore
      })
    }

    setBoard(newBoard)
    setCurrentPiece(null)
  }, [currentPiece, position, board, mergePieceToBoard, clearLines])

  const moveDown = useCallback(() => {
    if (!currentPiece || isPaused || gameOver) return

    const newPos = { x: position.x, y: position.y + 1 }
    if (checkCollision(currentPiece, newPos, board)) {
      lockPiece()
    } else {
      setPosition(newPos)
    }
  }, [currentPiece, position, board, isPaused, gameOver, checkCollision, lockPiece])

  const moveLeft = useCallback(() => {
    if (!currentPiece || isPaused || gameOver) return
    const newPos = { x: position.x - 1, y: position.y }
    if (!checkCollision(currentPiece, newPos, board)) {
      setPosition(newPos)
      setMoves(m => m + 1)
    }
  }, [currentPiece, position, board, isPaused, gameOver, checkCollision])

  const moveRight = useCallback(() => {
    if (!currentPiece || isPaused || gameOver) return
    const newPos = { x: position.x + 1, y: position.y }
    if (!checkCollision(currentPiece, newPos, board)) {
      setPosition(newPos)
      setMoves(m => m + 1)
    }
  }, [currentPiece, position, board, isPaused, gameOver, checkCollision])

  const rotatePiece = useCallback(() => {
    if (!currentPiece || isPaused || gameOver) return
    const rotated = { ...currentPiece, shape: rotate(currentPiece.shape) }
    if (!checkCollision(rotated, position, board)) {
      setCurrentPiece(rotated)
      setMoves(m => m + 1)
    }
  }, [currentPiece, position, board, isPaused, gameOver, checkCollision])

  const hardDrop = useCallback(() => {
    if (!currentPiece || isPaused || gameOver) return
    let newY = position.y
    while (!checkCollision(currentPiece, { x: position.x, y: newY + 1 }, board)) {
      newY++
    }
    setPosition({ x: position.x, y: newY })
    setMoves(m => m + 1)
  }, [currentPiece, position, board, isPaused, gameOver, checkCollision])

  const togglePause = useCallback(() => {
    setIsPaused(p => !p)
  }, [])

  const resetGame = useCallback(() => {
    setBoardWidth(initialWidth)
    setBoardHeight(initialHeight)
    setBoard(createEmptyBoard(initialWidth, initialHeight))
    setCurrentPiece(null)
    setPosition({ x: 0, y: 0 })
    setScore(0)
    setMoves(0)
    setLevel(1)
    setGameOver(false)
    setIsPaused(false)
    setElapsedTime(0)
    setIsFrozen(false)
    setFreezeTimeLeft(0)
  }, [initialWidth, initialHeight])

  const growBoard = useCallback(() => {
    setBoardWidth(w => {
      const newWidth = w + 1
      setBoard(prevBoard => {
        return prevBoard.map(row => [...row, null])
      })
      return newWidth
    })
    setBoardHeight(h => {
      const newHeight = h + 1
      setBoard(prevBoard => {
        return [Array(prevBoard[0].length).fill(null), ...prevBoard]
      })
      return newHeight
    })
  }, [])

  const triggerFreeze = useCallback(() => {
    if (isFrozen) return
    setIsFrozen(true)
    setFreezeTimeLeft(10)

    if (freezeRef.current) clearTimeout(freezeRef.current)
    if (freezeCountdownRef.current) clearInterval(freezeCountdownRef.current)

    freezeCountdownRef.current = setInterval(() => {
      setFreezeTimeLeft(t => {
        if (t <= 1) {
          clearInterval(freezeCountdownRef.current)
          return 0
        }
        return t - 1
      })
    }, 1000)

    freezeRef.current = setTimeout(() => {
      setIsFrozen(false)
      setFreezeTimeLeft(0)
    }, 10000)
  }, [isFrozen])

  useEffect(() => {
    if (!currentPiece && !gameOver && !isPaused) {
      spawnPiece()
    }
  }, [currentPiece, gameOver, isPaused, spawnPiece])

  useEffect(() => {
    if (gameOver || isPaused) {
      if (gameLoopRef.current) clearInterval(gameLoopRef.current)
      return
    }

    const speed = Math.max(100, 1000 - (level - 1) * 100)
    gameLoopRef.current = setInterval(() => {
      if (!isFrozen) {
        moveDown()
      }
    }, speed)

    return () => {
      if (gameLoopRef.current) clearInterval(gameLoopRef.current)
    }
  }, [moveDown, level, gameOver, isPaused, isFrozen])

  useEffect(() => {
    if (gameOver || isPaused) {
      if (timeRef.current) clearInterval(timeRef.current)
      return
    }

    timeRef.current = setInterval(() => {
      setElapsedTime(t => t + 1)
      if (Math.random() < 0.02) {
        triggerFreeze()
      }
    }, 1000)

    return () => {
      if (timeRef.current) clearInterval(timeRef.current)
    }
  }, [gameOver, isPaused, triggerFreeze])

  useEffect(() => {
    if (gameOver || isPaused) {
      if (growRef.current) clearInterval(growRef.current)
      return
    }

    growRef.current = setInterval(() => {
      growBoard()
    }, growInterval)

    return () => {
      if (growRef.current) clearInterval(growRef.current)
    }
  }, [gameOver, isPaused, growInterval, growBoard])

  useEffect(() => {
    return () => {
      if (freezeRef.current) clearTimeout(freezeRef.current)
      if (freezeCountdownRef.current) clearInterval(freezeCountdownRef.current)
    }
  }, [])

  const getDisplayBoard = useCallback(() => {
    if (!currentPiece) return board
    const displayBoard = board.map(row => [...row])
    for (let y = 0; y < currentPiece.shape.length; y++) {
      for (let x = 0; x < currentPiece.shape[y].length; x++) {
        if (currentPiece.shape[y][x]) {
          const boardY = position.y + y
          const boardX = position.x + x
          if (boardY >= 0 && boardY < displayBoard.length && boardX >= 0 && boardX < displayBoard[0].length) {
            displayBoard[boardY][boardX] = currentPiece.color
          }
        }
      }
    }
    return displayBoard
  }, [board, currentPiece, position])

  return {
    board: getDisplayBoard(),
    score,
    moves,
    level,
    gameOver,
    isPaused,
    elapsedTime,
    isFrozen,
    freezeTimeLeft,
    boardWidth,
    boardHeight,
    moveLeft,
    moveRight,
    moveDown,
    rotatePiece,
    hardDrop,
    togglePause,
    resetGame
  }
}
