import { useQuery } from '@tanstack/react-query'
import { fetchEngines } from '../api/client'

interface Props {
  value: string
  onChange: (v: string) => void
}

const selectStyle: React.CSSProperties = {
  padding: '8px 12px',
  fontSize: '13px',
  background: '#1a1a2e',
  color: '#e0e0e0',
  border: '1px solid #2a2a4a',
  borderRadius: '6px',
  outline: 'none',
  cursor: 'pointer',
}

export default function EngineSelector({ value, onChange }: Props) {
  const { data: engines, isLoading } = useQuery({
    queryKey: ['engines'],
    queryFn: fetchEngines,
  })

  if (isLoading) return <select style={selectStyle} disabled><option>Loading...</option></select>

  return (
    <select
      style={selectStyle}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      <option value="">Select engine</option>
      {engines?.map((e) => (
        <option key={e.id} value={e.id}>{e.name}</option>
      ))}
    </select>
  )
}
