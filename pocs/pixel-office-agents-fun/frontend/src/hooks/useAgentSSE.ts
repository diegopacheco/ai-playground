import { useState, useEffect, useRef } from 'react'
import { SSEEvent } from '../types'
import { getStreamUrl } from '../api/agents'

export function useAgentSSE(agentId: string | null) {
  const [status, setStatus] = useState<string>('idle')
  const [messages, setMessages] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    setStatus('idle')
    setMessages([])
    setError(null)
    if (!agentId) return

    const es = new EventSource(getStreamUrl(agentId))
    esRef.current = es

    es.addEventListener('agent_status', (e) => {
      const data: SSEEvent = JSON.parse(e.data)
      setStatus(data.status || 'unknown')
    })

    es.addEventListener('agent_message', (e) => {
      const data: SSEEvent = JSON.parse(e.data)
      if (data.content) {
        setMessages(prev => [...prev, data.content!])
      }
    })

    es.addEventListener('agent_done', (e) => {
      const data: SSEEvent = JSON.parse(e.data)
      setStatus(data.status || 'done')
      es.close()
    })

    es.addEventListener('agent_error', (e) => {
      const data: SSEEvent = JSON.parse(e.data)
      setError(data.message || 'Unknown error')
      setStatus('error')
      es.close()
    })

    es.onerror = () => {
      es.close()
    }

    return () => {
      es.close()
    }
  }, [agentId])

  return { status, messages, error }
}
