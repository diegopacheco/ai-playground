import { useState, useEffect, useCallback } from 'react'
import { createGameStream } from '../api/client'

interface GameState {
  board: string[][]
  currentPlayer: string
  status: 'playing' | 'finished'
  winner: string | null
  isDraw: boolean
  thinkingAgent: string | null
  lastMove: number | null
  durationMs: number | null
  error: string | null
}

const EMPTY_BOARD = Array(6).fill(null).map(() => Array(7).fill('.'))

export function useGameStream(gameId: string | null) {
  const [state, setState] = useState<GameState>({
    board: EMPTY_BOARD,
    currentPlayer: '',
    status: 'playing',
    winner: null,
    isDraw: false,
    thinkingAgent: null,
    lastMove: null,
    durationMs: null,
    error: null,
  })

  const reset = useCallback(() => {
    setState({
      board: EMPTY_BOARD,
      currentPlayer: '',
      status: 'playing',
      winner: null,
      isDraw: false,
      thinkingAgent: null,
      lastMove: null,
      durationMs: null,
      error: null,
    })
  }, [])

  useEffect(() => {
    if (!gameId) return
    reset()
    const eventSource = createGameStream(gameId)
    eventSource.addEventListener('board_update', (e) => {
      const data = JSON.parse(e.data)
      setState((prev) => ({
        ...prev,
        board: data.board,
        currentPlayer: data.current_player,
        lastMove: data.last_move,
        thinkingAgent: null,
      }))
    })
    eventSource.addEventListener('agent_thinking', (e) => {
      const data = JSON.parse(e.data)
      setState((prev) => ({
        ...prev,
        thinkingAgent: data.agent,
      }))
    })
    eventSource.addEventListener('agent_moved', () => {
      setState((prev) => ({
        ...prev,
        thinkingAgent: null,
      }))
    })
    eventSource.addEventListener('game_over', (e) => {
      const data = JSON.parse(e.data)
      setState((prev) => ({
        ...prev,
        status: 'finished',
        winner: data.winner,
        isDraw: data.is_draw,
        durationMs: data.duration_ms,
        thinkingAgent: null,
      }))
      eventSource.close()
    })
    eventSource.addEventListener('error', (e) => {
      if (e instanceof MessageEvent) {
        const data = JSON.parse(e.data)
        setState((prev) => ({
          ...prev,
          error: data.message,
        }))
      }
    })
    return () => {
      eventSource.close()
    }
  }, [gameId, reset])

  return state
}
