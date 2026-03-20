import type { WeeklyScore } from "../types";
import ClassificationBadge from "./ClassificationBadge";

interface WeeklyChartProps {
  scores: WeeklyScore[];
}

function WeeklyChart({ scores }: WeeklyChartProps) {
  const maxScore = Math.max(...scores.map((s) => s.total_score), 10);

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-5">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">
        Weekly Trend
      </h2>
      <div className="space-y-3">
        {scores.map((score) => {
          const width = Math.max((score.total_score / maxScore) * 100, 4);
          return (
            <div key={score.id} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-700 font-medium">
                  Week of {score.week_start}
                </span>
                <span className="text-gray-500">
                  {score.num_commits} commits | avg {score.avg_score.toFixed(1)}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                  <div
                    className="bg-blue-500 h-6 rounded-full flex items-center justify-end pr-2 transition-all"
                    style={{ width: `${width}%` }}
                  >
                    <span className="text-white text-xs font-bold">
                      {score.total_score}
                    </span>
                  </div>
                </div>
                <div className="flex gap-1.5 shrink-0">
                  <span className="flex items-center gap-0.5 text-xs">
                    <ClassificationBadge classification="DEEP" /> {score.deep_count}
                  </span>
                  <span className="flex items-center gap-0.5 text-xs">
                    <ClassificationBadge classification="DECENT" /> {score.decent_count}
                  </span>
                  <span className="flex items-center gap-0.5 text-xs">
                    <ClassificationBadge classification="SHALLOW" /> {score.shallow_count}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default WeeklyChart;
