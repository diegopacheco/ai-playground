import { useState } from 'react'
import { AGENT_TYPES, CreateAgentRequest } from '../types'

interface Props {
  onSpawn: (req: CreateAgentRequest) => void
}

export default function AgentSpawnForm({ onSpawn }: Props) {
  const [name, setName] = useState('')
  const [agentType, setAgentType] = useState('claude')
  const [task, setTask] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim() || !task.trim()) return
    onSpawn({ name: name.trim(), agent_type: agentType, task: task.trim() })
    setName('')
    setTask('')
  }

  return (
    <form onSubmit={handleSubmit} style={formStyle}>
      <div style={headerStyle}>Spawn Agent</div>
      <div style={fieldRow}>
        <input
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="Agent name"
          style={inputStyle}
        />
        <select value={agentType} onChange={e => setAgentType(e.target.value)} style={selectStyle}>
          {AGENT_TYPES.map(t => (
            <option key={t.id} value={t.id}>{t.name} ({t.model})</option>
          ))}
        </select>
      </div>
      <textarea
        value={task}
        onChange={e => setTask(e.target.value)}
        placeholder="What should this agent do?"
        rows={3}
        style={textareaStyle}
      />
      <button type="submit" style={buttonStyle}>
        Deploy to Office
      </button>
    </form>
  )
}

const formStyle: React.CSSProperties = {
  background: '#1a1a2e',
  border: '1px solid #333',
  borderRadius: 8,
  padding: 16,
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
}

const headerStyle: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 'bold',
  color: '#4ade80',
  textTransform: 'uppercase',
  letterSpacing: 2,
}

const fieldRow: React.CSSProperties = {
  display: 'flex',
  gap: 8,
}

const inputStyle: React.CSSProperties = {
  flex: 1,
  background: '#0f0f23',
  border: '1px solid #444',
  borderRadius: 4,
  padding: '8px 12px',
  color: '#e0e0e0',
  fontFamily: 'Courier New, monospace',
  fontSize: 13,
  outline: 'none',
}

const selectStyle: React.CSSProperties = {
  ...inputStyle,
  flex: 'none',
  width: 200,
  cursor: 'pointer',
}

const textareaStyle: React.CSSProperties = {
  ...inputStyle,
  resize: 'vertical',
  minHeight: 60,
}

const buttonStyle: React.CSSProperties = {
  background: '#4ade80',
  color: '#0f0f23',
  border: 'none',
  borderRadius: 4,
  padding: '10px 20px',
  fontFamily: 'Courier New, monospace',
  fontWeight: 'bold',
  fontSize: 13,
  cursor: 'pointer',
  textTransform: 'uppercase',
  letterSpacing: 1,
}
