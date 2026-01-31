import { useState } from 'react';
import { useDebates, useDebate } from '../hooks/useDebates';
import { DebateCard } from './DebateCard';
import { ChatWidget } from './ChatWidget';
import { getAgentInfo } from '../types';

interface HistoryListProps {
  onBack: () => void;
}

export function HistoryList({ onBack }: HistoryListProps) {
  const { data: debates = [] } = useDebates();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const { data: selectedDebate } = useDebate(selectedId);

  if (selectedDebate) {
    const agentAInfo = getAgentInfo(selectedDebate.agent_a);
    const agentBInfo = getAgentInfo(selectedDebate.agent_b);
    const judgeInfo = getAgentInfo(selectedDebate.agent_judge);
    const getWinnerName = (winner: string) => {
      if (winner === 'A') return agentAInfo.name;
      if (winner === 'B') return agentBInfo.name;
      return winner;
    };

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
            messages={selectedDebate.messages}
            thinkingAgent={null}
            agentAInfo={agentAInfo}
            agentBInfo={agentBInfo}
          />

          {selectedDebate.winner && (
            <div className="bg-gray-800 p-6 border-t border-gray-700">
              <div className="text-center">
                <div className="text-2xl font-bold text-white mb-2">
                  Winner: {getWinnerName(selectedDebate.winner)}
                </div>
                <p className="text-gray-300">{selectedDebate.judge_reason}</p>
                <p className="text-gray-500 text-sm mt-2">Judged by {judgeInfo.name}</p>
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
