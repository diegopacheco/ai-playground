import { useEffect, useState, useCallback } from 'react'
import { useTetris } from '../hooks/useTetris'

const TetrisBoard = () => {
  const [config, setConfig] = useState(null)
  const [configKey, setConfigKey] = useState(0)

  const loadConfig = useCallback(() => {
    fetch('/api/config')
      .then(res => res.json())
      .then(data => {
        if (data) {
          const newConfig = {
            boardWidth: data.board_width || 10,
            boardHeight: data.board_height || 20,
            growInterval: data.grow_interval || 30000,
            theme: data.theme || 'dark',
            dropSpeed: data.drop_speed || 1000,
            freezeChance: data.freeze_chance || 2
          }
          setConfig(newConfig)
          setConfigKey(prev => prev + 1)
        }
      })
      .catch(() => {
        setConfig({
          boardWidth: 10,
          boardHeight: 20,
          growInterval: 30000,
          theme: 'dark',
          dropSpeed: 1000,
          freezeChance: 2
        })
      })
  }, [])

  useEffect(() => {
    loadConfig()
  }, [loadConfig])

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        loadConfig()
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    window.addEventListener('focus', loadConfig)
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      window.removeEventListener('focus', loadConfig)
    }
  }, [loadConfig])

  if (!config) {
    return (
      <div className="tetris-container theme-dark">
        <div className="loading">Loading...</div>
      </div>
    )
  }

  return (
    <TetrisGame
      key={configKey}
      config={config}
    />
  )
}

const TetrisGame = ({ config }) => {
  const {
    board,
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
  } = useTetris(config.boardWidth, config.boardHeight, config.growInterval)

  useEffect(() => {
    const handleKeyDown = (e) => {
      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault()
          moveLeft()
          break
        case 'ArrowRight':
          e.preventDefault()
          moveRight()
          break
        case 'ArrowDown':
          e.preventDefault()
          moveDown()
          break
        case 'ArrowUp':
          e.preventDefault()
          rotatePiece()
          break
        case ' ':
          e.preventDefault()
          hardDrop()
          break
        case 'p':
        case 'P':
          e.preventDefault()
          togglePause()
          break
        case 'r':
        case 'R':
          e.preventDefault()
          if (gameOver) resetGame()
          break
        default:
          break
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [moveLeft, moveRight, moveDown, rotatePiece, hardDrop, togglePause, resetGame, gameOver])

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const themeClass = `tetris-container theme-${config.theme}`

  return (
    <div className={themeClass}>
      <div className="game-header">
        <h1>TETRIS</h1>
      </div>
      <div className="game-layout">
        <div className="stats-panel">
          <div className="stat-box">
            <span className="stat-label">SCORE</span>
            <span className="stat-value">{score}</span>
          </div>
          <div className="stat-box">
            <span className="stat-label">LEVEL</span>
            <span className="stat-value">{level}</span>
          </div>
          <div className="stat-box">
            <span className="stat-label">TIME</span>
            <span className="stat-value">{formatTime(elapsedTime)}</span>
          </div>
          <div className="stat-box">
            <span className="stat-label">MOVES</span>
            <span className="stat-value">{moves}</span>
          </div>
          <div className="stat-box">
            <span className="stat-label">BOARD</span>
            <span className="stat-value">{boardWidth}x{boardHeight}</span>
          </div>
        </div>
        <div className="board-container">
          {isFrozen && (
            <div className="freeze-overlay">
              <span>FREEZE BONUS!</span>
              <span className="freeze-timer">{freezeTimeLeft}s</span>
            </div>
          )}
          {(gameOver || isPaused) && (
            <div className="game-overlay">
              {gameOver ? (
                <>
                  <span className="overlay-title">GAME OVER</span>
                  <span className="overlay-subtitle">Press R to restart</span>
                </>
              ) : (
                <>
                  <span className="overlay-title">PAUSED</span>
                  <span className="overlay-subtitle">Press P to resume</span>
                </>
              )}
            </div>
          )}
          <div
            className="tetris-board"
            style={{
              gridTemplateColumns: `repeat(${board[0]?.length || 10}, 1fr)`,
              gridTemplateRows: `repeat(${board.length || 20}, 1fr)`
            }}
          >
            {board.map((row, y) =>
              row.map((cell, x) => (
                <div
                  key={`${y}-${x}`}
                  className={`cell ${cell ? 'filled' : 'empty'}`}
                  style={cell ? { backgroundColor: cell } : {}}
                />
              ))
            )}
          </div>
        </div>
        <div className="controls-panel">
          <h3>CONTROLS</h3>
          <div className="control-item">
            <span className="control-key">LEFT/RIGHT</span>
            <span>Move</span>
          </div>
          <div className="control-item">
            <span className="control-key">UP</span>
            <span>Rotate</span>
          </div>
          <div className="control-item">
            <span className="control-key">DOWN</span>
            <span>Soft Drop</span>
          </div>
          <div className="control-item">
            <span className="control-key">SPACE</span>
            <span>Hard Drop</span>
          </div>
          <div className="control-item">
            <span className="control-key">P</span>
            <span>Pause</span>
          </div>
          <div className="game-buttons">
            <button onClick={togglePause} className="game-btn">
              {isPaused ? 'RESUME' : 'PAUSE'}
            </button>
            <button onClick={resetGame} className="game-btn">
              RESTART
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TetrisBoard
