import type { Debate } from '../types';

interface DebateCardProps {
  debate: Debate;
  onClick: () => void;
}

export function DebateCard({ debate, onClick }: DebateCardProps) {
  return (
    <div
      onClick={onClick}
      className="bg-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-700 transition"
    >
      <h3 className="text-white font-bold text-lg mb-2 truncate">
        {debate.topic}
      </h3>
      <div className="flex items-center gap-4 text-sm text-gray-400">
        <span>{debate.agent_a} vs {debate.agent_b}</span>
        <span>Judge: {debate.agent_judge}</span>
      </div>
      {debate.winner && (
        <div className="mt-2 text-green-400">
          Winner: Agent {debate.winner}
        </div>
      )}
      <div className="text-xs text-gray-500 mt-2">
        {new Date(debate.started_at).toLocaleString()}
      </div>
    </div>
  );
}
