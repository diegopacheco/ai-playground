import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchStatus } from '../api'
import type { ClusterObject } from '../api'
import StatusBadge from '../components/StatusBadge'
import YamlModal from '../components/YamlModal'

const cellStyle: React.CSSProperties = { padding: '8px 12px', borderBottom: '1px solid #21262d', fontSize: 13 }
const headerStyle: React.CSSProperties = { ...cellStyle, fontWeight: 600, color: '#8b949e', textAlign: 'left', position: 'sticky' as const, top: 0, background: '#0d1117' }

export default function ClusterPage() {
  const [selected, setSelected] = useState<ClusterObject | null>(null)
  const { data, isLoading, error } = useQuery({ queryKey: ['status'], queryFn: fetchStatus, refetchInterval: 10000 })

  if (isLoading) return <p>Loading cluster status...</p>
  if (error) return <p style={{ color: '#da3633' }}>Error: {(error as Error).message}</p>

  const grouped: Record<string, ClusterObject[]> = {}
  for (const obj of data || []) {
    const key = obj.kind
    if (!grouped[key]) grouped[key] = []
    grouped[key].push(obj)
  }

  return (
    <div>
      {Object.entries(grouped).map(([kind, objects]) => (
        <div key={kind} style={{ marginBottom: 24 }}>
          <h2 style={{ fontSize: 16, color: '#58a6ff', marginBottom: 8 }}>{kind}s ({objects.length})</h2>
          <table style={{ width: '100%', borderCollapse: 'collapse', background: '#161b22', borderRadius: 8, overflow: 'hidden' }}>
            <thead>
              <tr>
                <th style={headerStyle}>Namespace</th>
                <th style={headerStyle}>Name</th>
                <th style={headerStyle}>Status</th>
                <th style={headerStyle}>Ready</th>
                <th style={headerStyle}>Restarts</th>
                <th style={headerStyle}>Age</th>
              </tr>
            </thead>
            <tbody>
              {objects.map(obj => (
                <tr
                  key={`${obj.namespace}/${obj.name}`}
                  onClick={() => setSelected(obj)}
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={e => (e.currentTarget.style.background = '#21262d')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                >
                  <td style={cellStyle}>{obj.namespace}</td>
                  <td style={{ ...cellStyle, color: '#58a6ff' }}>{obj.name}</td>
                  <td style={cellStyle}><StatusBadge status={obj.status} /></td>
                  <td style={cellStyle}>{obj.ready}</td>
                  <td style={cellStyle}>{obj.restarts}</td>
                  <td style={cellStyle}>{obj.age}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      {selected && (
        <YamlModal
          yamlContent={selected.yaml}
          title={`${selected.kind}: ${selected.namespace}/${selected.name}`}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  )
}
