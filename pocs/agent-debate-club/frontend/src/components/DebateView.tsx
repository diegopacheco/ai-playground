import { useDebateSSE } from '../hooks/useDebateSSE';
import { ChatWidget } from './ChatWidget';
import { Timer } from './Timer';
import { getAgentInfo } from '../types';

interface DebateViewProps {
  debateId: string;
  topic: string;
  duration: number;
  agentA: string;
  agentB: string;
  judge: string;
  onBack: () => void;
}

export function DebateView({ debateId, topic, duration, agentA, agentB, judge, onBack }: DebateViewProps) {
  const { messages, status, thinkingAgent, result, error } = useDebateSSE(debateId);
  const agentAInfo = getAgentInfo(agentA);
  const agentBInfo = getAgentInfo(agentB);
  const judgeInfo = getAgentInfo(judge);

  const isRunning = status === 'thinking' || (status === 'idle' && !result);

  const getWinnerName = (winner: string) => {
    if (winner === 'A') return agentAInfo.name;
    if (winner === 'B') return agentBInfo.name;
    return winner;
  };

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      <header className="bg-gray-800 p-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <button
            onClick={onBack}
            className="text-gray-400 hover:text-white transition"
          >
            Back
          </button>
          <h1 className="text-xl font-bold text-white flex-1 text-center px-4 truncate">
            {topic}
          </h1>
          <div className="text-white">
            <Timer durationSeconds={duration} isRunning={isRunning} />
          </div>
        </div>
        <div className="max-w-4xl mx-auto mt-2 flex items-center justify-center gap-4 text-sm">
          <span style={{ color: agentAInfo.color }}>{agentAInfo.name} ({agentAInfo.model})</span>
          <span className="text-gray-500">vs</span>
          <span style={{ color: agentBInfo.color }}>{agentBInfo.name} ({agentBInfo.model})</span>
          <span className="text-gray-500">|</span>
          <span className="text-gray-400">Judge: <span style={{ color: judgeInfo.color }}>{judgeInfo.name} ({judgeInfo.model})</span></span>
        </div>
      </header>

      <div className="flex-1 max-w-4xl mx-auto w-full flex flex-col">
        <ChatWidget
          messages={messages}
          thinkingAgent={thinkingAgent}
          agentAInfo={agentAInfo}
          agentBInfo={agentBInfo}
        />

        {error && (
          <div className="bg-red-600 text-white p-4 text-center">{error}</div>
        )}

        {result && (
          <div className="bg-gray-800 p-6 border-t border-gray-700">
            <div className="text-center">
              <div className="text-2xl font-bold text-white mb-2">
                Winner: {getWinnerName(result.winner)}
              </div>
              <p className="text-gray-300">{result.reason}</p>
              <p className="text-gray-500 text-sm mt-2">
                Duration: {Math.round(result.duration_ms / 1000)}s | Judged by {judgeInfo.name}
              </p>
              <button
                onClick={onBack}
                className="mt-4 px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
              >
                New Debate
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
