import { useDebateSSE } from '../hooks/useDebateSSE';
import { ChatWidget } from './ChatWidget';
import { Timer } from './Timer';

interface DebateViewProps {
  debateId: string;
  topic: string;
  duration: number;
  onBack: () => void;
}

export function DebateView({ debateId, topic, duration, onBack }: DebateViewProps) {
  const { messages, status, thinkingAgent, result, error } = useDebateSSE(debateId);

  const isRunning = status === 'thinking' || (status === 'idle' && !result);

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
      </header>

      <div className="flex-1 max-w-4xl mx-auto w-full flex flex-col">
        <ChatWidget messages={messages} thinkingAgent={thinkingAgent} />

        {error && (
          <div className="bg-red-600 text-white p-4 text-center">{error}</div>
        )}

        {result && (
          <div className="bg-gray-800 p-6 border-t border-gray-700">
            <div className="text-center">
              <div className="text-2xl font-bold text-white mb-2">
                Winner: Agent {result.winner}
              </div>
              <p className="text-gray-300">{result.reason}</p>
              <p className="text-gray-500 text-sm mt-2">
                Duration: {Math.round(result.duration_ms / 1000)}s
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
