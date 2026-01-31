import { useState } from 'react';
import { useAgents } from '../hooks/useDebates';
import { createDebate } from '../api/debates';
import { AGENT_INFO } from '../types';

interface ThemeSetupProps {
  onDebateStarted: (debateId: string, duration: number, topic: string, agentA: string, agentB: string, judge: string) => void;
}

export function ThemeSetup({ onDebateStarted }: ThemeSetupProps) {
  const { data: agents = [] } = useAgents();
  const [topic, setTopic] = useState('');
  const [agentA, setAgentA] = useState('claude');
  const [agentB, setAgentB] = useState('gemini');
  const [judge, setJudge] = useState('claude');
  const [duration, setDuration] = useState(60);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!topic.trim()) return;

    setIsLoading(true);
    try {
      const response = await createDebate({
        topic: topic.trim(),
        agent_a: agentA,
        agent_b: agentB,
        agent_judge: judge,
        duration_seconds: duration,
      });
      onDebateStarted(response.id, duration, topic.trim(), agentA, agentB, judge);
    } catch {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-lg p-8 w-full max-w-md">
        <h1 className="text-3xl font-bold text-white mb-8 text-center">
          Agent Debate Club
        </h1>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-gray-300 mb-2">Debate Topic</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Enter a topic to debate..."
              className="w-full px-4 py-3 rounded bg-gray-700 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-300 mb-2">Agent A</label>
              <select
                value={agentA}
                onChange={(e) => setAgentA(e.target.value)}
                className="w-full px-4 py-3 rounded bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {agents.map((agent) => (
                  <option key={agent.id} value={agent.id}>
                    {agent.name} ({AGENT_INFO[agent.id]?.model || 'unknown'})
                  </option>
                ))}
              </select>
              <div className="mt-1 text-sm text-gray-400">
                Model: <span style={{ color: AGENT_INFO[agentA]?.color }}>{AGENT_INFO[agentA]?.model}</span>
              </div>
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Agent B</label>
              <select
                value={agentB}
                onChange={(e) => setAgentB(e.target.value)}
                className="w-full px-4 py-3 rounded bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {agents.map((agent) => (
                  <option key={agent.id} value={agent.id}>
                    {agent.name} ({AGENT_INFO[agent.id]?.model || 'unknown'})
                  </option>
                ))}
              </select>
              <div className="mt-1 text-sm text-gray-400">
                Model: <span style={{ color: AGENT_INFO[agentB]?.color }}>{AGENT_INFO[agentB]?.model}</span>
              </div>
            </div>
          </div>

          <div>
            <label className="block text-gray-300 mb-2">Judge</label>
            <select
              value={judge}
              onChange={(e) => setJudge(e.target.value)}
              className="w-full px-4 py-3 rounded bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {agents.map((agent) => (
                <option key={agent.id} value={agent.id}>
                  {agent.name} ({AGENT_INFO[agent.id]?.model || 'unknown'})
                </option>
              ))}
            </select>
            <div className="mt-1 text-sm text-gray-400">
              Model: <span style={{ color: AGENT_INFO[judge]?.color }}>{AGENT_INFO[judge]?.model}</span>
            </div>
          </div>

          <div>
            <label className="block text-gray-300 mb-2">
              Duration (seconds): {duration}
            </label>
            <input
              type="range"
              min="30"
              max="300"
              step="10"
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <button
            type="submit"
            disabled={isLoading || !topic.trim()}
            className="w-full py-3 bg-blue-600 text-white rounded font-bold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {isLoading ? 'Starting...' : 'Start Debate'}
          </button>
        </form>
      </div>
    </div>
  );
}
