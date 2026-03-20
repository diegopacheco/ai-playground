import { useState } from "react";
import { useTrackUser, useUntrackUser, useLeaderboard } from "../hooks/useLeaderboard";
import { useAnalyze } from "../hooks/useAnalysis";

function LeaderboardForm() {
  const [value, setValue] = useState("");
  const trackMutation = useTrackUser();
  const untrackMutation = useUntrackUser();
  const analyzeMutation = useAnalyze();
  const leaderboardQuery = useLeaderboard();

  const trackedUsers = [
    ...new Set(leaderboardQuery.data?.map((e) => e.github_user) ?? []),
  ];

  const handleAdd = (e: React.FormEvent) => {
    e.preventDefault();
    const user = value.trim();
    if (!user) return;
    trackMutation.mutate(user, {
      onSuccess: () => {
        analyzeMutation.mutate(user);
        setValue("");
      },
    });
  };

  const handleRemove = (user: string) => {
    untrackMutation.mutate(user);
  };

  return (
    <div className="space-y-3">
      <form onSubmit={handleAdd} className="flex gap-3">
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="GitHub username"
          className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
        />
        <button
          type="submit"
          disabled={trackMutation.isPending || !value.trim()}
          className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white font-medium px-6 py-2 rounded-lg transition-colors cursor-pointer disabled:cursor-not-allowed"
        >
          Add &amp; Analyze
        </button>
      </form>
      {trackedUsers.length > 0 && (
        <div className="flex gap-2 flex-wrap">
          <span className="text-gray-500 text-sm py-1">Tracking:</span>
          {trackedUsers.map((user) => (
            <span
              key={user}
              className="inline-flex items-center gap-1 bg-gray-800 border border-gray-700 rounded-full px-3 py-1 text-sm text-gray-300"
            >
              {user}
              <button
                onClick={() => handleRemove(user)}
                className="text-gray-500 hover:text-red-400 transition-colors cursor-pointer ml-1"
              >
                &#x2715;
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default LeaderboardForm;
