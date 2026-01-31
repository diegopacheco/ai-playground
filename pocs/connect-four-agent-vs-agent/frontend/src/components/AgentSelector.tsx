import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchAgents } from '../api/client'
import { getAgentInfo } from '../types'

interface AgentSelectorProps {
  onStart: (agentA: string, agentB: string) => void
  disabled: boolean
}

export function AgentSelector({ onStart, disabled }: AgentSelectorProps) {
  const [agentA, setAgentA] = useState('')
  const [agentB, setAgentB] = useState('')
  const { data: agents, isLoading } = useQuery({
    queryKey: ['agents'],
    queryFn: fetchAgents,
  })

  const handleStart = () => {
    if (agentA && agentB) {
      onStart(agentA, agentB)
    }
  }

  const agentAInfo = agentA ? getAgentInfo(agentA) : null
  const agentBInfo = agentB ? getAgentInfo(agentB) : null

  if (isLoading) {
    return <div className="text-center p-8">Loading agents...</div>
  }

  return (
    <div className="flex flex-col gap-6 p-8 bg-gray-800 rounded-lg max-w-xl mx-auto">
      <h2 className="text-2xl font-bold text-center">Select Agents</h2>
      <div className="grid grid-cols-2 gap-8">
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-full bg-red-500" />
            <label className="text-lg font-semibold">Player X</label>
          </div>
          <select
            value={agentA}
            onChange={(e) => setAgentA(e.target.value)}
            disabled={disabled}
            className="bg-gray-700 text-white px-4 py-3 rounded border border-gray-600 focus:border-blue-500 focus:outline-none text-lg"
          >
            <option value="">Select Agent</option>
            {agents?.map((a) => {
              const info = getAgentInfo(a)
              return (
                <option key={a} value={a}>{info.name}</option>
              )
            })}
          </select>
          {agentAInfo && (
            <div className="bg-gray-700 rounded-lg p-4 border-l-4" style={{ borderColor: agentAInfo.color }}>
              <div className="text-lg font-bold" style={{ color: agentAInfo.color }}>{agentAInfo.name}</div>
              <div className="text-sm text-gray-400 mt-1">Model: <span className="text-white font-mono">{agentAInfo.model}</span></div>
            </div>
          )}
        </div>
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-full bg-yellow-400" />
            <label className="text-lg font-semibold">Player O</label>
          </div>
          <select
            value={agentB}
            onChange={(e) => setAgentB(e.target.value)}
            disabled={disabled}
            className="bg-gray-700 text-white px-4 py-3 rounded border border-gray-600 focus:border-blue-500 focus:outline-none text-lg"
          >
            <option value="">Select Agent</option>
            {agents?.map((a) => {
              const info = getAgentInfo(a)
              return (
                <option key={a} value={a}>{info.name}</option>
              )
            })}
          </select>
          {agentBInfo && (
            <div className="bg-gray-700 rounded-lg p-4 border-l-4" style={{ borderColor: agentBInfo.color }}>
              <div className="text-lg font-bold" style={{ color: agentBInfo.color }}>{agentBInfo.name}</div>
              <div className="text-sm text-gray-400 mt-1">Model: <span className="text-white font-mono">{agentBInfo.model}</span></div>
            </div>
          )}
        </div>
      </div>
      {agentA && agentB && (
        <div className="text-center text-gray-400 text-sm">
          {agentAInfo?.name} ({agentAInfo?.model}) vs {agentBInfo?.name} ({agentBInfo?.model})
        </div>
      )}
      <button
        onClick={handleStart}
        disabled={!agentA || !agentB || disabled}
        className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold px-6 py-4 rounded text-lg transition-colors"
      >
        {disabled ? 'Game in Progress...' : 'Start Game'}
      </button>
    </div>
  )
}
