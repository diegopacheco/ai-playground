import { QueryEvent } from '../types'

interface Props {
  events: QueryEvent[]
}

function EventLog({ events }: Props) {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <h3 className="text-sm font-semibold text-slate-400 mb-3">Agent Activity</h3>
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {events.map((event, i) => (
          <div key={i} className="flex items-start gap-2 text-sm">
            <span className="mt-0.5">{getIcon(event.type)}</span>
            <div className="flex-1">
              <span className={getColor(event.type)}>{getLabel(event.type)}</span>
              <span className="text-slate-400 ml-2">{getMessage(event)}</span>
              {('sql' in event && event.type !== 'QueryResult') && (
                <pre className="mt-1 bg-slate-900 rounded p-2 text-xs text-green-400 overflow-x-auto">
                  {(event as any).sql}
                </pre>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function getIcon(type: string): string {
  switch (type) {
    case 'Thinking': return '...'
    case 'SqlGenerated': return '[SQL]'
    case 'SqlError': return '[ERR]'
    case 'SqlFixed': return '[FIX]'
    case 'QueryResult': return '[OK]'
    case 'Failed': return '[FAIL]'
    default: return '[-]'
  }
}

function getColor(type: string): string {
  switch (type) {
    case 'Thinking': return 'text-yellow-400'
    case 'SqlGenerated': return 'text-cyan-400'
    case 'SqlError': return 'text-red-400'
    case 'SqlFixed': return 'text-orange-400'
    case 'QueryResult': return 'text-green-400'
    case 'Failed': return 'text-red-500'
    default: return 'text-slate-400'
  }
}

function getLabel(type: string): string {
  switch (type) {
    case 'Thinking': return 'Thinking'
    case 'SqlGenerated': return 'SQL Generated'
    case 'SqlError': return 'SQL Error'
    case 'SqlFixed': return 'SQL Fixed'
    case 'QueryResult': return 'Success'
    case 'Failed': return 'Failed'
    default: return type
  }
}

function getMessage(event: QueryEvent): string {
  switch (event.type) {
    case 'Thinking': return event.message
    case 'SqlGenerated': return `Attempt ${event.attempt}`
    case 'SqlError': return event.error
    case 'SqlFixed': return `Attempt ${event.attempt}`
    case 'QueryResult': return `${event.rows.length} rows returned`
    case 'Failed': return event.error
    default: return ''
  }
}

export default EventLog
