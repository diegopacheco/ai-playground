import type { Analysis } from "../types";
import ClassificationBadge from "./ClassificationBadge";

interface ScoreSummaryProps {
  analysis: Analysis;
}

function ScoreSummary({ analysis }: ScoreSummaryProps) {
  const deep = analysis.commits.filter(
    (c) => c.classification === "DEEP"
  ).length;
  const decent = analysis.commits.filter(
    (c) => c.classification === "DECENT"
  ).length;
  const shallow = analysis.commits.filter(
    (c) => c.classification === "SHALLOW"
  ).length;

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-5">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">
        Score Summary
      </h2>
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
        <div>
          <div className="text-gray-500 text-xs uppercase tracking-wide">
            Total
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {analysis.total_score ?? 0}
            <span className="text-gray-500 text-sm font-normal">/100</span>
          </div>
        </div>
        <div>
          <div className="text-gray-500 text-xs uppercase tracking-wide">
            Average
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {(analysis.avg_score ?? 0).toFixed(1)}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <ClassificationBadge classification="DEEP" />
          <span className="text-gray-900 font-medium">x{deep}</span>
        </div>
        <div className="flex items-center gap-2">
          <ClassificationBadge classification="DECENT" />
          <span className="text-gray-900 font-medium">x{decent}</span>
        </div>
        <div className="flex items-center gap-2">
          <ClassificationBadge classification="SHALLOW" />
          <span className="text-gray-900 font-medium">x{shallow}</span>
        </div>
      </div>
    </div>
  );
}

export default ScoreSummary;
