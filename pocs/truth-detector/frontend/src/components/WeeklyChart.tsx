import type { WeeklyScore } from "../types";

interface WeeklyChartProps {
  scores: WeeklyScore[];
}

function WeeklyChart({ scores }: WeeklyChartProps) {
  const maxScore = Math.max(...scores.map((s) => s.total_score), 1);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-5">
      <h2 className="text-lg font-semibold text-gray-200 mb-4">
        Weekly Trend
      </h2>
      <div className="flex items-end gap-2 h-40">
        {scores.map((score) => {
          const height = (score.total_score / maxScore) * 100;
          return (
            <div
              key={score.id}
              className="flex flex-col items-center flex-1 gap-1"
            >
              <span className="text-xs text-gray-400">
                {score.total_score}
              </span>
              <div
                className="w-full bg-blue-500 rounded-t transition-all"
                style={{ height: `${height}%` }}
              />
              <span className="text-xs text-gray-500 truncate w-full text-center">
                {score.week_start.substring(5)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default WeeklyChart;
