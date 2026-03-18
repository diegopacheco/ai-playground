import { SpanRecord, STEP_COLORS, STEP_LABELS } from '../types'

interface Props {
  spans: SpanRecord[]
  activeStep: string | null
}

export default function SpanTimeline({ spans, activeStep }: Props) {
  const childSpans = spans.filter(s => s.parent_span_id !== null)
  const rootSpan = spans.find(s => s.parent_span_id === null)

  if (childSpans.length === 0 && !activeStep) {
    return (
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <h2 className="text-xl font-bold mb-4 text-cyan-400">Span Timeline</h2>
        <p className="text-slate-500">No spans yet. Run an agent to see the trace.</p>
      </div>
    )
  }

  const rootStart = rootSpan?.start_time_ns ?? (childSpans[0]?.start_time_ns ?? 0)
  const allEndTimes = childSpans
    .map(s => s.end_time_ns ?? 0)
    .filter(t => t > 0)
  const maxEnd = allEndTimes.length > 0 ? Math.max(...allEndTimes) : rootStart + 1_000_000_000
  const totalRange = maxEnd - rootStart

  const allSteps = ['analyze', 'plan', 'research', 'synthesize']
  const pendingSteps = allSteps.filter(
    step => !childSpans.find(s => s.name === step) && step !== activeStep
  )

  return (
    <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
      <h2 className="text-xl font-bold mb-4 text-cyan-400">Span Timeline (Waterfall)</h2>

      <div className="space-y-3">
        {childSpans.map((span) => {
          const offset = totalRange > 0
            ? ((span.start_time_ns - rootStart) / totalRange) * 100
            : 0
          const width = totalRange > 0 && span.end_time_ns
            ? ((span.end_time_ns - span.start_time_ns) / totalRange) * 100
            : 5
          const color = STEP_COLORS[span.name] ?? '#64748b'
          const tokens = span.attributes['tokens.total'] ?? '0'

          return (
            <div key={span.span_id} className="flex items-center gap-3">
              <div className="w-24 text-right text-sm text-slate-400 shrink-0">
                {STEP_LABELS[span.name] ?? span.name}
              </div>
              <div className="flex-1 bg-slate-900 rounded h-10 relative overflow-hidden">
                <div
                  className="absolute h-full rounded flex items-center px-2 text-xs font-bold text-white transition-all duration-500"
                  style={{
                    left: `${offset}%`,
                    width: `${Math.max(width, 3)}%`,
                    backgroundColor: color,
                  }}
                >
                  <span className="truncate">
                    {span.duration_ms ? `${(span.duration_ms / 1000).toFixed(1)}s` : '...'}
                    {' | '}
                    {tokens} tok
                  </span>
                </div>
              </div>
            </div>
          )
        })}

        {activeStep && (
          <div className="flex items-center gap-3">
            <div className="w-24 text-right text-sm text-slate-400 shrink-0">
              {STEP_LABELS[activeStep] ?? activeStep}
            </div>
            <div className="flex-1 bg-slate-900 rounded h-10 relative overflow-hidden">
              <div
                className="absolute h-full rounded flex items-center px-3 text-xs font-bold text-white animate-pulse"
                style={{
                  left: '0%',
                  width: '100%',
                  backgroundColor: STEP_COLORS[activeStep] ?? '#64748b',
                  opacity: 0.6,
                }}
              >
                thinking...
              </div>
            </div>
          </div>
        )}

        {pendingSteps
          .filter(step => step !== activeStep)
          .map(step => (
            <div key={step} className="flex items-center gap-3 opacity-30">
              <div className="w-24 text-right text-sm text-slate-500 shrink-0">
                {STEP_LABELS[step] ?? step}
              </div>
              <div className="flex-1 bg-slate-900 rounded h-10 relative overflow-hidden">
                <div
                  className="absolute h-full rounded border border-dashed"
                  style={{
                    left: '0%',
                    width: '100%',
                    borderColor: STEP_COLORS[step] ?? '#64748b',
                  }}
                />
              </div>
            </div>
          ))}
      </div>

      {rootSpan && rootSpan.duration_ms && (
        <div className="mt-4 pt-4 border-t border-slate-700 flex gap-6 text-sm">
          <div>
            <span className="text-slate-400">Total Duration: </span>
            <span className="text-white font-bold">
              {(rootSpan.duration_ms / 1000).toFixed(1)}s
            </span>
          </div>
          <div>
            <span className="text-slate-400">Total Tokens: </span>
            <span className="text-white font-bold">
              {rootSpan.attributes['tokens.total'] ?? '0'}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
