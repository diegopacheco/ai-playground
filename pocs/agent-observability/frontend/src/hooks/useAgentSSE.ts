import { useEffect, useRef } from 'react'
import { SSEEvent } from '../types'

export function useAgentSSE(
  traceId: string | null,
  onEvent: (event: SSEEvent) => void
) {
  const eventSourceRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!traceId) return

    const es = new EventSource(
      `http://localhost:3000/api/traces/${traceId}/stream`
    )
    eventSourceRef.current = es

    es.onmessage = (e) => {
      const data: SSEEvent = JSON.parse(e.data)
      onEvent(data)
    }

    es.onerror = () => {
      es.close()
    }

    return () => {
      es.close()
      eventSourceRef.current = null
    }
  }, [traceId])
}
