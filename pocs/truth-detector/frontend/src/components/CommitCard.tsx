import type { CommitResult } from "../types";
import ClassificationBadge from "./ClassificationBadge";

interface CommitCardProps {
  commit: CommitResult;
}

function CommitCard({ commit }: CommitCardProps) {
  const commitUrl = `https://github.com/${commit.repo_name}/commit/${commit.commit_sha}`;

  return (
    <a
      href={commitUrl}
      target="_blank"
      rel="noopener noreferrer"
      className="block bg-gray-50 border border-gray-200 rounded-lg p-4 space-y-2 hover:border-blue-400 hover:bg-blue-50/30 transition-colors cursor-pointer"
    >
      <div className="flex items-center gap-3 flex-wrap">
        <ClassificationBadge classification={commit.classification} />
        <span className="text-gray-600 text-sm">{commit.repo_name}</span>
        <span className="text-blue-600 text-xs font-mono underline">
          {commit.commit_sha.substring(0, 7)}
        </span>
        {commit.fallback && (
          <span className="text-xs bg-gray-200 text-gray-500 border border-gray-300 rounded px-1.5 py-0.5">
            fallback
          </span>
        )}
      </div>
      <p className="text-gray-800 text-sm">{commit.summary}</p>
      <div className="text-sm text-gray-500">
        Score: <span className="text-gray-900 font-medium">{commit.score}/10</span>
      </div>
    </a>
  );
}

export default CommitCard;
