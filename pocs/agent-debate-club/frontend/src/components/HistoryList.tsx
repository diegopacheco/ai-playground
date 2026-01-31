import { useState } from 'react';
import { useDebates, useDebate } from '../hooks/useDebates';
import { DebateCard } from './DebateCard';
import { ChatWidget } from './ChatWidget';

interface HistoryListProps {
  onBack: () => void;
}

export function HistoryList({ onBack }: HistoryListProps) {
  const { data: debates = [] } = useDebates();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const { data: selectedDebate } = useDebate(selectedId);

  if (selectedDebate) {
    return (
      <div className="min-h-screen bg-gray-900 flex flex-col">
        <header className="bg-gray-800 p-4">
          <div className="max-w-4xl mx-auto flex items-center">
            <button
              onClick={() => setSelectedId(null)}
              className="text-gray-400 hover:text-white transition"
            >
              Back to History
            </button>
            <h1 className="text-xl font-bold text-white flex-1 text-center px-4 truncate">
              {selectedDebate.topic}
            </h1>
          </div>
        </header>

        <div className="flex-1 max-w-4xl mx-auto w-full flex flex-col">
          <ChatWidget messages={selectedDebate.messages} thinkingAgent={null} />

          {selectedDebate.winner && (
            <div className="bg-gray-800 p-6 border-t border-gray-700">
              <div className="text-center">
                <div className="text-2xl font-bold text-white mb-2">
                  Winner: Agent {selectedDebate.winner}
                </div>
                <p className="text-gray-300">{selectedDebate.judge_reason}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <button
            onClick={onBack}
            className="text-gray-400 hover:text-white transition"
          >
            Back
          </button>
          <h1 className="text-2xl font-bold text-white">Debate History</h1>
          <div className="w-12" />
        </div>

        {debates.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            No debates yet. Start one!
          </div>
        ) : (
          <div className="space-y-4">
            {debates.map((debate) => (
              <DebateCard
                key={debate.id}
                debate={debate}
                onClick={() => setSelectedId(debate.id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
