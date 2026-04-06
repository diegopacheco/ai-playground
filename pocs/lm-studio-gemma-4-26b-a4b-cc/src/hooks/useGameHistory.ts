import { useState, useEffect } from 'react'
import type { GameResult } from '../logic/gameLogic'

export function useGameHistory() {
  const [history, setHistory] = useState<GameResult[]>([])

  useEffect(() => {
    const saved = localStorage.getItem('game-history')
    if (saved) {
      try {
        setHistory(JSON.parse(saved))
      } catch (e) {
        console.error('Failed to parse history', e)
      }
    }
  }, [])

  const addResult = (result: GameResult) => {
    setHistory((prev) => {
      const newHistory = [result, ...prev]
      localStorage.setItem('game-history', JSON.stringify(newHistory))
      return newHistory
    })
  }

  return { history, addResult }
}
