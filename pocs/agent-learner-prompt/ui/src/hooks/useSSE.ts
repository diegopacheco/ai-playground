import { useEffect, useState, useCallback } from 'react'
import type { ProgressEvent } from '../types'

export function useSSE(taskId: string | null) {
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [connected, setConnected] = useState(false)

  const clear = useCallback(() => setEvents([]), [])

  useEffect(() => {
    if (!taskId) {
      setConnected(false)
      return
    }
    const es = new EventSource(`/api/tasks/${taskId}/events`)
    es.onopen = () => setConnected(true)
    es.onmessage = (e) => {
      try {
        const event: ProgressEvent = JSON.parse(e.data)
        setEvents((prev) => [...prev, event])
      } catch {
        console.error('Failed to parse SSE event')
      }
    }
    es.onerror = () => {
      setConnected(false)
      es.close()
    }
    return () => {
      es.close()
      setConnected(false)
    }
  }, [taskId])

  return { events, connected, clear }
}
