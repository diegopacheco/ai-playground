import { TraceRecord, STEP_COLORS, STEP_LABELS } from '../types'

interface Props {
  trace: TraceRecord
}

export default function TraceView({ trace }: Props) {
  const childSpans = trace.spans.filter(s => s.parent_span_id !== null)

  const totalInputTokens = childSpans.reduce(
    (sum, s) => sum + parseInt(s.attributes['tokens.input'] ?? '0'), 0
  )
  const totalOutputTokens = childSpans.reduce(
    (sum, s) => sum + parseInt(s.attributes['tokens.output'] ?? '0'), 0
  )

  return (
    <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-cyan-400">Trace Details</h2>
        <a
          href={`http://localhost:16686/search?service=agent-observability`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-amber-400 hover:text-amber-300 underline"
        >
          Open in Jaeger
        </a>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-900 rounded p-3">
          <div className="text-xs text-slate-400">Status</div>
          <div className={`text-lg font-bold ${trace.status === 'completed' ? 'text-green-400' : 'text-amber-400'}`}>
            {trace.status}
          </div>
        </div>
        <div className="bg-slate-900 rounded p-3">
          <div className="text-xs text-slate-400">Duration</div>
          <div className="text-lg font-bold text-white">
            {(trace.total_duration_ms / 1000).toFixed(1)}s
          </div>
        </div>
        <div className="bg-slate-900 rounded p-3">
          <div className="text-xs text-slate-400">Input Tokens</div>
          <div className="text-lg font-bold text-blue-400">{totalInputTokens}</div>
        </div>
        <div className="bg-slate-900 rounded p-3">
          <div className="text-xs text-slate-400">Output Tokens</div>
          <div className="text-lg font-bold text-purple-400">{totalOutputTokens}</div>
        </div>
      </div>

      <div className="mb-6">
        <h3 className="text-sm text-slate-400 mb-2">Token Distribution</h3>
        <div className="flex h-8 rounded overflow-hidden">
          {childSpans.map(span => {
            const tokens = parseInt(span.attributes['tokens.total'] ?? '0')
            const pct = trace.total_tokens_estimated > 0
              ? (tokens / trace.total_tokens_estimated) * 100
              : 25
            return (
              <div
                key={span.span_id}
                className="flex items-center justify-center text-xs font-bold text-white"
                style={{
                  width: `${pct}%`,
                  backgroundColor: STEP_COLORS[span.name] ?? '#64748b',
                }}
                title={`${STEP_LABELS[span.name] ?? span.name}: ${tokens} tokens`}
              >
                {STEP_LABELS[span.name] ?? span.name}
              </div>
            )
          })}
        </div>
      </div>

      <div className="space-y-3">
        <h3 className="text-sm text-slate-400">Step Details</h3>
        {childSpans.map(span => (
          <div
            key={span.span_id}
            className="bg-slate-900 rounded p-4 border-l-4"
            style={{ borderColor: STEP_COLORS[span.name] ?? '#64748b' }}
          >
            <div className="flex justify-between items-center mb-2">
              <span className="font-bold text-white">
                {STEP_LABELS[span.name] ?? span.name}
              </span>
              <div className="flex gap-4 text-xs text-slate-400">
                <span>{span.duration_ms ? `${(span.duration_ms / 1000).toFixed(1)}s` : '-'}</span>
                <span>{span.attributes['tokens.total'] ?? '0'} tokens</span>
                <span className={span.status === 'ok' ? 'text-green-400' : 'text-red-400'}>
                  {span.status}
                </span>
              </div>
            </div>
            <div className="text-sm text-slate-300 whitespace-pre-wrap break-words max-h-40 overflow-y-auto">
              {span.attributes['output.preview'] ?? 'No output'}
            </div>
          </div>
        ))}
      </div>

      {trace.result && (
        <div className="mt-6">
          <h3 className="text-sm text-slate-400 mb-2">Final Result</h3>
          <div className="bg-slate-900 rounded p-4 text-sm text-slate-200 whitespace-pre-wrap break-words max-h-60 overflow-y-auto border border-cyan-800">
            {trace.result}
          </div>
        </div>
      )}

      <div className="mt-4 text-xs text-slate-500">
        Trace ID: {trace.trace_id} | Agent: {trace.agent} | Started: {trace.started_at}
      </div>
    </div>
  )
}
