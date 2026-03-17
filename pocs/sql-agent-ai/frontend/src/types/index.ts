export interface QueryRecord {
  id: string
  question: string
  status: string
  generated_sql: string | null
  result: string | null
  created_at: string
}

export interface CreateQueryRequest {
  question: string
}

export interface CreateQueryResponse {
  id: string
}

export interface ThinkingEvent {
  type: 'Thinking'
  message: string
}

export interface SqlGeneratedEvent {
  type: 'SqlGenerated'
  sql: string
  attempt: number
}

export interface SqlErrorEvent {
  type: 'SqlError'
  error: string
  attempt: number
}

export interface SqlFixedEvent {
  type: 'SqlFixed'
  sql: string
  attempt: number
}

export interface QueryResultEvent {
  type: 'QueryResult'
  columns: string[]
  rows: (string | number | boolean | null)[][]
  sql: string
}

export interface FailedEvent {
  type: 'Failed'
  error: string
}

export type QueryEvent =
  | ThinkingEvent
  | SqlGeneratedEvent
  | SqlErrorEvent
  | SqlFixedEvent
  | QueryResultEvent
  | FailedEvent

export interface TableInfo {
  name: string
  columns: { name: string; data_type: string }[]
}
