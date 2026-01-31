import type { Debate } from '../types';
import { getAgentInfo } from '../types';

interface DebateCardProps {
  debate: Debate;
  onClick: () => void;
}

export function DebateCard({ debate, onClick }: DebateCardProps) {
  const agentAInfo = getAgentInfo(debate.agent_a);
  const agentBInfo = getAgentInfo(debate.agent_b);
  const judgeInfo = getAgentInfo(debate.agent_judge);
  const getWinnerName = (winner: string) => {
    if (winner === 'A') return agentAInfo.name;
    if (winner === 'B') return agentBInfo.name;
    return winner;
  };

  return (
    <div
      onClick={onClick}
      className="bg-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-700 transition"
    >
      <h3 className="text-white font-bold text-lg mb-2 truncate">
        {debate.topic}
      </h3>
      <div className="flex items-center gap-4 text-sm">
        <span style={{ color: agentAInfo.color }}>{agentAInfo.name} ({agentAInfo.model})</span>
        <span className="text-gray-500">vs</span>
        <span style={{ color: agentBInfo.color }}>{agentBInfo.name} ({agentBInfo.model})</span>
      </div>
      <div className="text-sm text-gray-400 mt-1">
        Judge: <span style={{ color: judgeInfo.color }}>{judgeInfo.name} ({judgeInfo.model})</span>
      </div>
      {debate.winner && (
        <div className="mt-2 text-green-400">
          Winner: {getWinnerName(debate.winner)}
        </div>
      )}
      <div className="text-xs text-gray-500 mt-2">
        {new Date(debate.started_at).toLocaleString()}
      </div>
    </div>
  );
}
