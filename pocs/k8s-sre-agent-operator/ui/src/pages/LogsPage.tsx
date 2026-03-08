import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchLogs } from '../api'

export default function LogsPage() {
  const [autoRefresh, setAutoRefresh] = useState(true)

  const { data, isLoading, error } = useQuery({
    queryKey: ['logs'],
    queryFn: fetchLogs,
    refetchInterval: autoRefresh ? 5000 : false,
  })

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h2 style={{ fontSize: 16, color: '#58a6ff' }}>Pod Logs</h2>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 14, color: '#8b949e' }}>
          Auto-refresh (5s)
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={e => setAutoRefresh(e.target.checked)}
            style={{ cursor: 'pointer' }}
          />
          <span style={{ padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600, background: autoRefresh ? '#238636' : '#30363d', color: '#fff' }}>
            {autoRefresh ? 'ON' : 'OFF'}
          </span>
        </label>
      </div>
      {isLoading && <p>Loading logs...</p>}
      {error && <p style={{ color: '#da3633' }}>Error: {(error as Error).message}</p>}
      {data && (
        <pre style={{
          background: '#161b22',
          border: '1px solid #30363d',
          borderRadius: 8,
          padding: 16,
          fontSize: 12,
          fontFamily: 'monospace',
          overflow: 'auto',
          maxHeight: 'calc(100vh - 160px)',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          lineHeight: 1.5,
        }}>
          {data}
        </pre>
      )}
    </div>
  )
}
