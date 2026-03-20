import type { CommitResult } from "../types";
import ClassificationBadge from "./ClassificationBadge";

interface CommitCardProps {
  commit: CommitResult;
}

function CommitCard({ commit }: CommitCardProps) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-2 hover:border-gray-700 transition-colors">
      <div className="flex items-center gap-3 flex-wrap">
        <ClassificationBadge classification={commit.classification} />
        <span className="text-gray-300 text-sm">{commit.repo_name}</span>
        <span className="text-gray-500 text-xs font-mono">
          {commit.commit_sha.substring(0, 7)}
        </span>
        {commit.fallback && (
          <span className="text-xs bg-gray-800 text-gray-500 border border-gray-700 rounded px-1.5 py-0.5">
            fallback
          </span>
        )}
      </div>
      <p className="text-gray-200 text-sm">{commit.summary}</p>
      <div className="text-sm text-gray-400">
        Score: <span className="text-white font-medium">{commit.score}/10</span>
      </div>
    </div>
  );
}

export default CommitCard;
