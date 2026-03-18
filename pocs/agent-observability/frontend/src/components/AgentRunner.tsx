import { useState } from 'react'
import { runAgent } from '../api/agent'

interface Props {
  onTraceStarted: (traceId: string) => void
}

export default function AgentRunner({ onTraceStarted }: Props) {
  const [topic, setTopic] = useState('')
  const [agent, setAgent] = useState('claude')
  const [loading, setLoading] = useState(false)

  const handleRun = async () => {
    if (!topic.trim()) return
    setLoading(true)
    const res = await runAgent(topic, agent)
    onTraceStarted(res.trace_id)
    setLoading(false)
  }

  return (
    <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
      <h2 className="text-xl font-bold mb-4 text-cyan-400">Run Agent</h2>
      <div className="space-y-4">
        <div>
          <label className="block text-sm text-slate-400 mb-1">Topic</label>
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="Enter a topic for the agent to analyze..."
            className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-slate-200 focus:border-cyan-500 focus:outline-none"
            onKeyDown={(e) => e.key === 'Enter' && handleRun()}
          />
        </div>
        <div className="flex gap-4 items-end">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Agent</label>
            <select
              value={agent}
              onChange={(e) => setAgent(e.target.value)}
              className="bg-slate-900 border border-slate-600 rounded px-3 py-2 text-slate-200 focus:border-cyan-500 focus:outline-none"
            >
              <option value="claude">Claude</option>
              <option value="gemini">Gemini</option>
            </select>
          </div>
          <button
            onClick={handleRun}
            disabled={loading || !topic.trim()}
            className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-600 text-white font-bold px-6 py-2 rounded transition-colors"
          >
            {loading ? 'Starting...' : 'Run'}
          </button>
        </div>
      </div>
    </div>
  )
}
