import { useState, useEffect, useRef } from 'react'
import { Agent, AGENT_TYPES } from '../types'
import { chatWithAgent, fetchAgent } from '../api/agents'

interface Props {
  agent: Agent
  onClose: () => void
}

export default function AgentChatPanel({ agent, onClose }: Props) {
  const [input, setInput] = useState('')
  const [chatHistory, setChatHistory] = useState<{ role: string; content: string }[]>([])
  const [thinking, setThinking] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const typeInfo = AGENT_TYPES.find(t => t.id === agent.agent_type)

  useEffect(() => {
    fetchAgent(agent.id).then(data => {
      if (data && data.messages && data.messages.length > 0) {
        setChatHistory(data.messages.map((m: any) => ({ role: m.role, content: m.content })))
      }
    }).catch(() => {})
  }, [agent.id])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [chatHistory, thinking])

  const handleSend = async () => {
    if (!input.trim() || thinking) return
    const msg = input.trim()
    setChatHistory(prev => [...prev, { role: 'user', content: msg }])
    setInput('')
    setThinking(true)
    try {
      const response = await chatWithAgent(agent.id, msg)
      setChatHistory(prev => [...prev, { role: response.role, content: response.content }])
    } catch {
      setChatHistory(prev => [...prev, { role: 'assistant', content: 'Failed to get response' }])
    }
    setThinking(false)
  }

  return (
    <div style={panelStyle}>
      <div style={headerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: typeInfo?.color || '#888', fontWeight: 'bold' }}>
            Chat: {agent.name}
          </span>
        </div>
        <button onClick={onClose} style={closeBtnStyle}>X</button>
      </div>
      <div ref={scrollRef} style={chatAreaStyle}>
        {chatHistory.length === 0 && !thinking && (
          <div style={{ color: '#555', fontStyle: 'italic', textAlign: 'center', padding: 20 }}>
            Start a conversation with {agent.name}
          </div>
        )}
        {chatHistory.map((msg, i) => (
          <div key={i} style={{
            ...msgBubble,
            alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
            background: msg.role === 'user' ? '#2a4a6a' : '#2a2a4a',
          }}>
            {msg.content}
          </div>
        ))}
        {thinking && (
          <div style={{ ...msgBubble, alignSelf: 'flex-start', background: '#2a2a4a', color: '#888' }}>
            Thinking...
          </div>
        )}
      </div>
      <div style={inputRowStyle}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Type a message..."
          style={chatInputStyle}
          disabled={thinking}
        />
        <button onClick={handleSend} style={{ ...sendBtnStyle, opacity: thinking ? 0.5 : 1 }} disabled={thinking}>
          {thinking ? '...' : 'Send'}
        </button>
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

const chatAreaStyle: React.CSSProperties = {
  flex: 1,
  padding: 14,
  overflowY: 'auto',
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
  minHeight: 200,
}

const msgBubble: React.CSSProperties = {
  padding: '8px 12px',
  borderRadius: 8,
  fontSize: 12,
  maxWidth: '80%',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
}

const inputRowStyle: React.CSSProperties = {
  display: 'flex',
  gap: 8,
  padding: 10,
  borderTop: '1px solid #333',
}

const chatInputStyle: React.CSSProperties = {
  flex: 1,
  background: '#0f0f23',
  border: '1px solid #444',
  borderRadius: 4,
  padding: '8px 12px',
  color: '#e0e0e0',
  fontFamily: 'Courier New, monospace',
  fontSize: 12,
  outline: 'none',
}

const sendBtnStyle: React.CSSProperties = {
  background: '#4ade80',
  color: '#0f0f23',
  border: 'none',
  borderRadius: 4,
  padding: '8px 16px',
  fontFamily: 'Courier New, monospace',
  fontWeight: 'bold',
  fontSize: 12,
  cursor: 'pointer',
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
