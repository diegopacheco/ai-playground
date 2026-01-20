import { Agent } from '../api/client'

interface AgentTabsProps {
  agents: Agent[]
  selectedAgent: string
  onSelectAgent: (name: string) => void
}

function AgentTabs({ agents, selectedAgent, onSelectAgent }: AgentTabsProps) {
  return (
    <div className="flex border-b border-slate-700">
      {agents.map((agent) => (
        <button
          key={agent.name}
          onClick={() => onSelectAgent(agent.name)}
          className={`px-6 py-3 font-medium transition-colors ${
            selectedAgent === agent.name
              ? 'text-blue-400 border-b-2 border-blue-400 bg-slate-800'
              : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
          }`}
        >
          {agent.name}
        </button>
      ))}
    </div>
  )
}

export default AgentTabs
