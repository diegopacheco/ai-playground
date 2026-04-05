import { useState, useEffect } from 'react'
import { useAgentSSE } from '../hooks/useAgentSSE'
import { fetchAgent } from '../api/agents'
import { Agent, AGENT_TYPES } from '../types'

interface Props {
  agent: Agent
  onClose: () => void
}

export default function AgentOutputPanel({ agent, onClose }: Props) {
  const isFinished = agent.status === 'done' || agent.status === 'error' || agent.status === 'stopped'
  const { status, messages: sseMessages, error } = useAgentSSE(isFinished ? null : agent.id)
  const [storedMessages, setStoredMessages] = useState<string[]>([])

  useEffect(() => {
    if (isFinished) {
      fetchAgent(agent.id).then(data => {
        if (data && data.messages) {
          setStoredMessages(data.messages.map((m: any) => m.content))
        }
      }).catch(() => {})
    } else {
      setStoredMessages([])
    }
  }, [agent.id, isFinished])

  const messages = isFinished ? storedMessages : sseMessages
  const typeInfo = AGENT_TYPES.find(t => t.id === agent.agent_type)

  return (
    <div style={panelStyle}>
      <div style={headerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: typeInfo?.color || '#888', fontWeight: 'bold' }}>
            {agent.name}
          </span>
          <span style={badgeStyle(agent.status)}>{agent.status}</span>
        </div>
        <button onClick={onClose} style={closeBtnStyle}>X</button>
      </div>
      <div style={taskStyle}>Task: {agent.task}</div>
      <div style={outputStyle}>
        {messages.length === 0 && !error && (
          <div style={{ color: '#666', fontStyle: 'italic' }}>
            {status === 'working' || status === 'thinking' ? 'Agent is working...' : 'No output yet'}
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} style={messageStyle}>{msg}</div>
        ))}
        {error && <div style={errorStyle}>{error}</div>}
      </div>
    </div>
  )
}

const panelStyle: React.CSSProperties = {
  background: '#1a1a2e',
  border: '1px solid #333',
  borderRadius: 8,
  display: 'flex',
  flexDirection: 'column',
  maxHeight: 400,
  overflow: 'hidden',
}

const headerStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: '10px 14px',
  borderBottom: '1px solid #333',
}

const taskStyle: React.CSSProperties = {
  padding: '8px 14px',
  fontSize: 11,
  color: '#888',
  borderBottom: '1px solid #222',
}

const outputStyle: React.CSSProperties = {
  flex: 1,
  padding: 14,
  overflowY: 'auto',
  fontSize: 12,
  lineHeight: 1.6,
}

const messageStyle: React.CSSProperties = {
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  marginBottom: 8,
}

const errorStyle: React.CSSProperties = {
  color: '#ff4444',
  fontWeight: 'bold',
}

const closeBtnStyle: React.CSSProperties = {
  background: 'none',
  border: '1px solid #555',
  color: '#888',
  borderRadius: 4,
  padding: '2px 8px',
  cursor: 'pointer',
  fontFamily: 'Courier New, monospace',
  fontSize: 11,
}

function badgeStyle(status: string): React.CSSProperties {
  const colors: Record<string, string> = {
    spawning: '#888',
    thinking: '#ffaa00',
    working: '#00cc44',
    done: '#4488ff',
    error: '#ff4444',
    stopped: '#888',
  }
  return {
    fontSize: 10,
    padding: '2px 6px',
    borderRadius: 3,
    background: colors[status] || '#888',
    color: '#000',
    fontWeight: 'bold',
    textTransform: 'uppercase',
  }
}
