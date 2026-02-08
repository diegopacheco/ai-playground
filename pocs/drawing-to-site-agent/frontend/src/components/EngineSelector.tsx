import { useQuery } from '@tanstack/react-query'
import { fetchEngines } from '../api/client'

interface Props {
  value: string
  onChange: (v: string) => void
}

const selectStyle: React.CSSProperties = {
  width: '100%',
  padding: '12px 16px',
  fontSize: '16px',
  background: '#1a1a2e',
  color: '#e0e0e0',
  border: '1px solid #2a2a4a',
  borderRadius: '8px',
  outline: 'none',
  cursor: 'pointer',
}

export default function EngineSelector({ value, onChange }: Props) {
  const { data: engines, isLoading } = useQuery({
    queryKey: ['engines'],
    queryFn: fetchEngines,
  })

  if (isLoading) return <select style={selectStyle} disabled><option>Loading engines...</option></select>

  return (
    <select
      style={selectStyle}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      <option value="">Select an engine</option>
      {engines?.map((e) => (
        <option key={e.id} value={e.id}>{e.name} ({e.id})</option>
      ))}
    </select>
  )
}
