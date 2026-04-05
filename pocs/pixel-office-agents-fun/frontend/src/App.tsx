import { useState, useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import PixelOffice from './components/PixelOffice'
import AgentSpawnForm from './components/AgentSpawnForm'
import AgentOutputPanel from './components/AgentOutputPanel'
import AgentChatPanel from './components/AgentChatPanel'
import { useAgents } from './hooks/useAgents'
import { spawnAgent, clearAllAgents } from './api/agents'
import { Agent, CreateAgentRequest } from './types'

export default function App() {
  const { data: agents = [] } = useAgents()
  const queryClient = useQueryClient()
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [panelMode, setPanelMode] = useState<'output' | 'chat' | null>(null)

  const handleSpawn = async (req: CreateAgentRequest) => {
    await spawnAgent(req)
    queryClient.invalidateQueries({ queryKey: ['agents'] })
  }

  const handleClear = async () => {
    await clearAllAgents()
    setSelectedAgent(null)
    setPanelMode(null)
    queryClient.invalidateQueries({ queryKey: ['agents'] })
  }

  const handleAgentClick = useCallback((agentId: string, clicks: number) => {
    const agent = agents.find(a => a.id === agentId)
    if (!agent) return
    setSelectedAgent(agent)
    setPanelMode(clicks === 2 ? 'chat' : 'output')
  }, [agents])

  const handleClosePanel = () => {
    setSelectedAgent(null)
    setPanelMode(null)
  }

  return (
    <div style={containerStyle}>
      <header style={headerStyle}>
        <h1 style={titleStyle}>Multi-Agent System</h1>
        <span style={subtitleStyle}>Pixel Office Control Panel</span>
      </header>
      <div style={mainStyle}>
        <div style={canvasCol}>
          <PixelOffice agents={agents} onAgentClick={handleAgentClick} />
          <div style={legendStyle}>
            <span style={legendItem}><span style={dot('#ffaa00')}></span> Thinking</span>
            <span style={legendItem}><span style={dot('#00cc44')}></span> Working</span>
            <span style={legendItem}><span style={dot('#4488ff')}></span> Done</span>
            <span style={legendItem}><span style={dot('#ff4444')}></span> Error</span>
            <span style={{ color: '#666', fontSize: 11 }}>Click = logs | Double-click = chat</span>
          </div>
        </div>
        <div style={sideCol}>
          <AgentSpawnForm onSpawn={handleSpawn} />
          {selectedAgent && panelMode === 'output' && (
            <AgentOutputPanel agent={selectedAgent} onClose={handleClosePanel} />
          )}
          {selectedAgent && panelMode === 'chat' && (
            <AgentChatPanel agent={selectedAgent} onClose={handleClosePanel} />
          )}
          <div style={agentListStyle}>
            <div style={listHeaderRow}>
              <span style={listHeaderText}>Active Agents ({agents.filter(a => a.status !== 'done' && a.status !== 'error').length})</span>
              {agents.length > 0 && (
                <button onClick={handleClear} style={clearBtnStyle}>Clear All</button>
              )}
            </div>
            {agents.map(a => (
              <div
                key={a.id}
                onClick={() => { setSelectedAgent(a); setPanelMode('output') }}
                onDoubleClick={() => { setSelectedAgent(a); setPanelMode('chat') }}
                style={agentRow(a.status)}
              >
                <span style={{ fontWeight: 'bold' }}>{a.name}</span>
                <span style={{ fontSize: 10, color: '#888' }}>{a.agent_type}</span>
                <span style={statusBadge(a.status)}>{a.status}</span>
              </div>
            ))}
            {agents.length === 0 && (
              <div style={{ padding: 12, color: '#555', fontSize: 12, textAlign: 'center' }}>
                No agents yet. Spawn one above.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

const containerStyle: React.CSSProperties = {
  minHeight: '100vh',
  display: 'flex',
  flexDirection: 'column',
  padding: 20,
  gap: 16,
  maxWidth: 1500,
  margin: '0 auto',
}

const headerStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'baseline',
  gap: 12,
}

const titleStyle: React.CSSProperties = {
  fontSize: 28,
  fontWeight: 'bold',
  color: '#4ade80',
  letterSpacing: 4,
  margin: 0,
}

const subtitleStyle: React.CSSProperties = {
  fontSize: 14,
  color: '#666',
  letterSpacing: 2,
}

const mainStyle: React.CSSProperties = {
  display: 'flex',
  gap: 20,
  flex: 1,
}

const canvasCol: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
}

const sideCol: React.CSSProperties = {
  width: 440,
  minWidth: 440,
  display: 'flex',
  flexDirection: 'column',
  gap: 12,
}

const legendStyle: React.CSSProperties = {
  display: 'flex',
  gap: 16,
  alignItems: 'center',
  fontSize: 11,
  color: '#aaa',
}

const legendItem: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 4,
}

function dot(color: string): React.CSSProperties {
  return {
    display: 'inline-block',
    width: 8,
    height: 8,
    borderRadius: '50%',
    background: color,
  }
}

const agentListStyle: React.CSSProperties = {
  background: '#1a1a2e',
  border: '1px solid #333',
  borderRadius: 8,
  overflow: 'hidden',
}

const listHeaderRow: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: '10px 14px',
  borderBottom: '1px solid #333',
}

const listHeaderText: React.CSSProperties = {
  fontSize: 12,
  fontWeight: 'bold',
  color: '#4ade80',
  textTransform: 'uppercase',
  letterSpacing: 1,
}

const clearBtnStyle: React.CSSProperties = {
  background: 'none',
  border: '1px solid #ff4444',
  color: '#ff4444',
  borderRadius: 4,
  padding: '3px 10px',
  cursor: 'pointer',
  fontFamily: 'Courier New, monospace',
  fontSize: 10,
  fontWeight: 'bold',
  textTransform: 'uppercase',
}

function agentRow(status: string): React.CSSProperties {
  return {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 14px',
    borderBottom: '1px solid #222',
    cursor: 'pointer',
    fontSize: 12,
    transition: 'background 0.15s',
  }
}

function statusBadge(status: string): React.CSSProperties {
  const colors: Record<string, string> = {
    spawning: '#888',
    thinking: '#ffaa00',
    working: '#00cc44',
    done: '#4488ff',
    error: '#ff4444',
    stopped: '#888',
  }
  return {
    fontSize: 9,
    padding: '2px 6px',
    borderRadius: 3,
    background: colors[status] || '#888',
    color: '#000',
    fontWeight: 'bold',
    textTransform: 'uppercase',
  }
}
