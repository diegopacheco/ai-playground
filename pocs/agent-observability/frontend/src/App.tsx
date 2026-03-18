import { useState, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import AgentRunner from './components/AgentRunner'
import SpanTimeline from './components/SpanTimeline'
import TraceView from './components/TraceView'
import { useAgentSSE } from './hooks/useAgentSSE'
import { getTraces, getTrace } from './api/agent'
import { SpanRecord, TraceRecord, SSEEvent } from './types'

export default function App() {
  const [activeTraceId, setActiveTraceId] = useState<string | null>(null)
  const [liveSpans, setLiveSpans] = useState<SpanRecord[]>([])
  const [activeStep, setActiveStep] = useState<string | null>(null)
  const [completedTrace, setCompletedTrace] = useState<TraceRecord | null>(null)

  const { data: traces, refetch: refetchTraces } = useQuery({
    queryKey: ['traces'],
    queryFn: getTraces,
    refetchInterval: 5000,
  })

  const handleSSEEvent = useCallback((event: SSEEvent) => {
    if (event.type === 'step_started') {
      setActiveStep(event.step ?? null)
    } else if (event.type === 'step_completed' && event.span) {
      setLiveSpans(prev => [...prev, event.span!])
      setActiveStep(null)
    } else if (event.type === 'trace_completed') {
      setActiveStep(null)
      refetchTraces()
      if (activeTraceId) {
        getTrace(activeTraceId).then(t => {
          if (t) setCompletedTrace(t)
        })
      }
    }
  }, [activeTraceId, refetchTraces])

  useAgentSSE(activeTraceId, handleSSEEvent)

  const handleTraceStarted = (traceId: string) => {
    setActiveTraceId(traceId)
    setLiveSpans([])
    setActiveStep(null)
    setCompletedTrace(null)
  }

  const handleSelectTrace = async (traceId: string) => {
    setActiveTraceId(traceId)
    setActiveStep(null)
    const t = await getTrace(traceId)
    if (t) {
      setLiveSpans(t.spans)
      if (t.status === 'completed') {
        setCompletedTrace(t)
      }
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-cyan-400">Agent Observability Stack</h1>
          <p className="text-slate-400 mt-1">
            Trace every agent decision as OpenTelemetry spans
          </p>
        </div>

        <AgentRunner onTraceStarted={handleTraceStarted} />

        <SpanTimeline spans={liveSpans} activeStep={activeStep} />

        {completedTrace && <TraceView trace={completedTrace} />}

        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-cyan-400">Trace History</h2>
            <a
              href="http://localhost:16686"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-amber-400 hover:text-amber-300 underline"
            >
              Jaeger UI
            </a>
          </div>
          {(!traces || traces.length === 0) ? (
            <p className="text-slate-500">No traces yet.</p>
          ) : (
            <div className="space-y-2">
              {traces.map(t => (
                <div
                  key={t.trace_id}
                  onClick={() => handleSelectTrace(t.trace_id)}
                  className={`bg-slate-900 rounded p-3 cursor-pointer hover:bg-slate-700 transition-colors border ${
                    activeTraceId === t.trace_id ? 'border-cyan-500' : 'border-slate-700'
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-white font-medium truncate">{t.topic}</div>
                      <div className="text-xs text-slate-400 mt-1">
                        {t.agent} | {t.spans.length} spans | {t.trace_id.substring(0, 12)}...
                      </div>
                    </div>
                    <div className="text-right ml-4 shrink-0">
                      <div className={`text-xs font-bold ${t.status === 'completed' ? 'text-green-400' : 'text-amber-400'}`}>
                        {t.status}
                      </div>
                      {t.total_duration_ms > 0 && (
                        <div className="text-xs text-slate-400">
                          {(t.total_duration_ms / 1000).toFixed(1)}s | ~{t.total_tokens_estimated} tok
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
