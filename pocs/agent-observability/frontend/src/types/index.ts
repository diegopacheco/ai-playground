export interface SpanRecord {
  trace_id: string
  span_id: string
  parent_span_id: string | null
  name: string
  start_time_ns: number
  end_time_ns: number | null
  duration_ms: number | null
  attributes: Record<string, string>
  status: string
}

export interface TraceRecord {
  trace_id: string
  topic: string
  agent: string
  started_at: string
  finished_at: string | null
  spans: SpanRecord[]
  result: string | null
  total_tokens_estimated: number
  total_duration_ms: number
  status: string
}

export interface RunResponse {
  trace_id: string
  status: string
}

export interface SSEEvent {
  type: 'step_started' | 'step_completed' | 'trace_completed'
  trace_id: string
  step?: string
  span_id?: string
  span?: SpanRecord
  total_duration_ms?: number
  total_tokens?: number
}

export const STEP_COLORS: Record<string, string> = {
  'analyze': '#3b82f6',
  'plan': '#10b981',
  'research': '#f59e0b',
  'synthesize': '#8b5cf6',
  'agent-run': '#64748b',
}

export const STEP_LABELS: Record<string, string> = {
  'analyze': 'Analyze',
  'plan': 'Plan',
  'research': 'Research',
  'synthesize': 'Synthesize',
}
