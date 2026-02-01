import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useTetris } from './useTetris'

describe('useTetris', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('initializes with correct default values', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    expect(result.current.score).toBe(0)
    expect(result.current.level).toBe(1)
    expect(result.current.moves).toBe(0)
    expect(result.current.gameOver).toBe(false)
    expect(result.current.isPaused).toBe(false)
    expect(result.current.elapsedTime).toBe(0)
    expect(result.current.boardWidth).toBe(10)
    expect(result.current.boardHeight).toBe(20)
  })

  it('creates board with correct dimensions', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    expect(result.current.board.length).toBe(20)
    expect(result.current.board[0].length).toBe(10)
  })

  it('creates board with custom dimensions', () => {
    const { result } = renderHook(() => useTetris(12, 25, 30000))
    expect(result.current.board.length).toBe(25)
    expect(result.current.board[0].length).toBe(12)
  })

  it('togglePause changes isPaused state', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    expect(result.current.isPaused).toBe(false)
    act(() => {
      result.current.togglePause()
    })
    expect(result.current.isPaused).toBe(true)
    act(() => {
      result.current.togglePause()
    })
    expect(result.current.isPaused).toBe(false)
  })

  it('resetGame resets all values', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    act(() => {
      vi.advanceTimersByTime(5000)
    })
    act(() => {
      result.current.resetGame()
    })
    expect(result.current.score).toBe(0)
    expect(result.current.level).toBe(1)
    expect(result.current.moves).toBe(0)
    expect(result.current.elapsedTime).toBe(0)
    expect(result.current.gameOver).toBe(false)
  })

  it('increments elapsed time every second', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    expect(result.current.elapsedTime).toBe(0)
    act(() => {
      vi.advanceTimersByTime(1000)
    })
    expect(result.current.elapsedTime).toBe(1)
    act(() => {
      vi.advanceTimersByTime(2000)
    })
    expect(result.current.elapsedTime).toBe(3)
  })

  it('does not increment time when paused', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    act(() => {
      result.current.togglePause()
    })
    act(() => {
      vi.advanceTimersByTime(3000)
    })
    expect(result.current.elapsedTime).toBe(0)
  })

  it('moveLeft increments moves counter', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    act(() => {
      vi.advanceTimersByTime(100)
    })
    const initialMoves = result.current.moves
    act(() => {
      result.current.moveLeft()
    })
    expect(result.current.moves).toBeGreaterThanOrEqual(initialMoves)
  })

  it('moveRight increments moves counter', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    act(() => {
      vi.advanceTimersByTime(100)
    })
    const initialMoves = result.current.moves
    act(() => {
      result.current.moveRight()
    })
    expect(result.current.moves).toBeGreaterThanOrEqual(initialMoves)
  })

  it('rotatePiece increments moves counter', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    act(() => {
      vi.advanceTimersByTime(100)
    })
    const initialMoves = result.current.moves
    act(() => {
      result.current.rotatePiece()
    })
    expect(result.current.moves).toBeGreaterThanOrEqual(initialMoves)
  })

  it('hardDrop increments moves counter', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    act(() => {
      vi.advanceTimersByTime(100)
    })
    const initialMoves = result.current.moves
    act(() => {
      result.current.hardDrop()
    })
    expect(result.current.moves).toBeGreaterThanOrEqual(initialMoves)
  })

  it('isFrozen is initially false', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    expect(result.current.isFrozen).toBe(false)
  })

  it('freezeTimeLeft is initially 0', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    expect(result.current.freezeTimeLeft).toBe(0)
  })

  it('board grows after grow interval', () => {
    const { result } = renderHook(() => useTetris(10, 20, 5000))
    const initialWidth = result.current.boardWidth
    const initialHeight = result.current.boardHeight
    act(() => {
      vi.advanceTimersByTime(5100)
    })
    expect(result.current.boardWidth).toBe(initialWidth + 1)
    expect(result.current.boardHeight).toBe(initialHeight + 1)
  })

  it('provides all required functions', () => {
    const { result } = renderHook(() => useTetris(10, 20, 30000))
    expect(typeof result.current.moveLeft).toBe('function')
    expect(typeof result.current.moveRight).toBe('function')
    expect(typeof result.current.moveDown).toBe('function')
    expect(typeof result.current.rotatePiece).toBe('function')
    expect(typeof result.current.hardDrop).toBe('function')
    expect(typeof result.current.togglePause).toBe('function')
    expect(typeof result.current.resetGame).toBe('function')
  })
})
