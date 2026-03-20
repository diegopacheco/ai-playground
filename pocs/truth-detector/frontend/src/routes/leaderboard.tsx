import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { useLeaderboard, useLeaderboardWeek } from "../hooks/useLeaderboard";
import LeaderboardForm from "../components/LeaderboardForm";
import LeaderboardTable from "../components/LeaderboardTable";

export const Route = createFileRoute("/leaderboard")({
  component: LeaderboardPage,
});

function getCurrentWeekStart(): string {
  const now = new Date();
  const day = now.getDay();
  const diff = now.getDate() - day + (day === 0 ? -6 : 1);
  const monday = new Date(now.setDate(diff));
  return monday.toISOString().split("T")[0];
}

function LeaderboardPage() {
  const [showPrevious, setShowPrevious] = useState(false);
  const leaderboardQuery = useLeaderboard();
  const weekStart = getCurrentWeekStart();

  const previousWeekStart = (() => {
    const d = new Date(weekStart);
    d.setDate(d.getDate() - 7);
    return d.toISOString().split("T")[0];
  })();

  const previousWeekQuery = useLeaderboardWeek(
    showPrevious ? previousWeekStart : null
  );

  const currentEnd = (() => {
    const d = new Date(weekStart);
    d.setDate(d.getDate() + 6);
    return d.toISOString().split("T")[0];
  })();

  const previousEnd = (() => {
    const d = new Date(previousWeekStart);
    d.setDate(d.getDate() + 6);
    return d.toISOString().split("T")[0];
  })();

  return (
    <div className="space-y-6">
      <LeaderboardForm />

      <div>
        <h2 className="text-lg font-semibold text-gray-800 mb-3">
          This Week ({weekStart} to {currentEnd})
        </h2>
        {leaderboardQuery.isLoading && (
          <p className="text-gray-500">Loading leaderboard...</p>
        )}
        {leaderboardQuery.data && (
          <LeaderboardTable entries={leaderboardQuery.data} />
        )}
        {leaderboardQuery.data?.length === 0 && (
          <p className="text-gray-500">
            No leaderboard data yet. Add and analyze users to populate.
          </p>
        )}
      </div>

      <div>
        <button
          onClick={() => setShowPrevious(!showPrevious)}
          className="text-sm text-blue-400 hover:text-blue-300 transition-colors cursor-pointer"
        >
          {showPrevious ? "Hide previous weeks" : "Show previous weeks"}
        </button>

        {showPrevious && (
          <div className="mt-4">
            <h2 className="text-lg font-semibold text-gray-800 mb-3">
              Previous Week ({previousWeekStart} to {previousEnd})
            </h2>
            {previousWeekQuery.isLoading && (
              <p className="text-gray-500">Loading...</p>
            )}
            {previousWeekQuery.data && (
              <LeaderboardTable entries={previousWeekQuery.data} />
            )}
            {previousWeekQuery.data?.length === 0 && (
              <p className="text-gray-500">No data for this week.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
