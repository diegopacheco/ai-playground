import { Agent } from '../api/client'

interface AgentCardProps {
  agent: Agent
  isSelected: boolean
  onClick: () => void
}

function AgentCard({ agent, isSelected, onClick }: AgentCardProps) {
  const getStatusStyles = () => {
    switch (agent.status) {
      case 'done':
        return 'border-green-500 bg-green-900/20'
      case 'running':
        return 'border-blue-500 bg-blue-900/20'
      case 'error':
        return 'border-red-500 bg-red-900/20'
      case 'timeout':
        return 'border-yellow-500 bg-yellow-900/20'
      default:
        return 'border-slate-600 bg-slate-800'
    }
  }

  const getStatusIcon = () => {
    switch (agent.status) {
      case 'done':
        return <span className="text-green-500 text-2xl">&#10003;</span>
      case 'running':
        return (
          <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        )
      case 'error':
        return <span className="text-red-500 text-2xl">&#10007;</span>
      case 'timeout':
        return <span className="text-yellow-500 text-2xl">&#8987;</span>
      default:
        return <span className="text-slate-500 text-2xl">&#9675;</span>
    }
  }

  return (
    <div
      onClick={onClick}
      className={`p-4 rounded-lg border-2 cursor-pointer transition-all hover:scale-105 ${getStatusStyles()} ${isSelected ? 'ring-2 ring-white' : ''}`}
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-bold text-lg">{agent.name}</h3>
        {getStatusIcon()}
      </div>
      <p className="text-sm text-slate-400">{agent.model}</p>
      <p className="text-xs text-slate-500 mt-1 capitalize">{agent.status}</p>
    </div>
  )
}

export default AgentCard
