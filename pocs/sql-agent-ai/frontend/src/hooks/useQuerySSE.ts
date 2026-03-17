import { useState, useEffect, useCallback } from 'react'
import { QueryEvent, QueryResultEvent } from '../types'
import { getStreamUrl } from '../api/queries'

interface SSEState {
  events: QueryEvent[]
  result: QueryResultEvent | null
  error: string | null
  isRunning: boolean
}

export function useQuerySSE(queryId: string | null) {
  const [state, setState] = useState<SSEState>({
    events: [],
    result: null,
    error: null,
    isRunning: false,
  })

  useEffect(() => {
    if (!queryId) return

    setState({ events: [], result: null, error: null, isRunning: true })

    const es = new EventSource(getStreamUrl(queryId))

    const handleEvent = (eventType: string) => (e: MessageEvent) => {
      const data = JSON.parse(e.data) as QueryEvent
      setState(prev => {
        const newEvents = [...prev.events, data]
        if (eventType === 'query_result') {
          return { ...prev, events: newEvents, result: data as QueryResultEvent, isRunning: false }
        }
        if (eventType === 'failed') {
          return { ...prev, events: newEvents, error: (data as any).error, isRunning: false }
        }
        return { ...prev, events: newEvents }
      })
      if (eventType === 'query_result' || eventType === 'failed') {
        es.close()
      }
    }

    es.addEventListener('thinking', handleEvent('thinking'))
    es.addEventListener('sql_generated', handleEvent('sql_generated'))
    es.addEventListener('sql_error', handleEvent('sql_error'))
    es.addEventListener('sql_fixed', handleEvent('sql_fixed'))
    es.addEventListener('query_result', handleEvent('query_result'))
    es.addEventListener('failed', handleEvent('failed'))

    es.onerror = () => {
      setState(prev => ({ ...prev, isRunning: false }))
      es.close()
    }

    return () => es.close()
  }, [queryId])

  const reset = useCallback(() => {
    setState({ events: [], result: null, error: null, isRunning: false })
  }, [])

  return { ...state, reset }
}
