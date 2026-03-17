import { useEffect, useRef } from 'react'
import { streamUrl } from '../api/games'

interface GameEvent {
  type: string
  text?: string
  message?: string
  reason?: string
}

export function useGameSSE(
  gameId: string,
  onNarration: (text: string) => void,
  onThinking: () => void,
  onError: (msg: string) => void,
) {
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!gameId) return

    const es = new EventSource(streamUrl(gameId))
    esRef.current = es

    es.addEventListener('dm_narration', (e) => {
      const data: GameEvent = JSON.parse(e.data)
      if (data.text) onNarration(data.text)
    })

    es.addEventListener('dm_thinking', () => {
      onThinking()
    })

    es.addEventListener('game_over', (e) => {
      const data: GameEvent = JSON.parse(e.data)
      onError(data.reason || 'Game Over')
      es.close()
    })

    es.addEventListener('error', (e) => {
      if (e instanceof MessageEvent) {
        const data: GameEvent = JSON.parse(e.data)
        onError(data.message || 'Unknown error')
      }
    })

    return () => {
      es.close()
    }
  }, [gameId])
}
