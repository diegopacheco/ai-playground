import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchHistory } from '../api'
import type { HistoryEvent } from '../api'

const cellStyle: React.CSSProperties = { padding: '8px 12px', borderBottom: '1px solid #21262d', fontSize: 13 }

export default function HistoryPage() {
  const [expanded, setExpanded] = useState<number | null>(null)
  const { data, isLoading, error } = useQuery({ queryKey: ['history'], queryFn: fetchHistory, refetchInterval: 5000 })

  if (isLoading) return <p>Loading history...</p>
  if (error) return <p style={{ color: '#da3633' }}>Error: {(error as Error).message}</p>

  const events = data || []

  if (events.length === 0) {
    return (
      <div>
        <h2 style={{ fontSize: 16, color: '#58a6ff', marginBottom: 16 }}>History</h2>
        <p style={{ color: '#8b949e' }}>No events recorded yet. Run a fix or apply to see history.</p>
      </div>
    )
  }

  return (
    <div>
      <h2 style={{ fontSize: 16, color: '#58a6ff', marginBottom: 16 }}>History ({events.length} events)</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse', background: '#161b22', borderRadius: 8, overflow: 'hidden' }}>
        <thead>
          <tr>
            <th style={{ ...cellStyle, fontWeight: 600, color: '#8b949e', textAlign: 'left' }}>Timestamp</th>
            <th style={{ ...cellStyle, fontWeight: 600, color: '#8b949e', textAlign: 'left' }}>Type</th>
            <th style={{ ...cellStyle, fontWeight: 600, color: '#8b949e', textAlign: 'left' }}>Summary</th>
            <th style={{ ...cellStyle, fontWeight: 600, color: '#8b949e', textAlign: 'left' }}>Status</th>
          </tr>
        </thead>
        <tbody>
          {events.map((evt: HistoryEvent, i: number) => (
            <>
              <tr
                key={i}
                onClick={() => setExpanded(expanded === i ? null : i)}
                style={{ cursor: 'pointer' }}
                onMouseEnter={e => (e.currentTarget.style.background = '#21262d')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
              >
                <td style={cellStyle}>{new Date(evt.timestamp).toLocaleString()}</td>
                <td style={cellStyle}>
                  <span style={{ padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600, background: evt.event_type === 'fix' ? '#1f3a5f' : '#2d333b', color: evt.event_type === 'fix' ? '#58a6ff' : '#c9d1d9' }}>
                    {evt.event_type}
                  </span>
                </td>
                <td style={{ ...cellStyle, maxWidth: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{evt.summary}</td>
                <td style={cellStyle}>
                  <span style={{ padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600, background: evt.success ? '#238636' : '#da3633', color: '#fff' }}>
                    {evt.success ? 'OK' : 'FAIL'}
                  </span>
                </td>
              </tr>
              {expanded === i && (
                <tr key={`${i}-details`}>
                  <td colSpan={4} style={{ padding: 0 }}>
                    <pre style={{ background: '#0d1117', padding: 16, fontSize: 12, fontFamily: 'monospace', overflow: 'auto', maxHeight: 300, whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0 }}>
                      {evt.details}
                    </pre>
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>
    </div>
  )
}
