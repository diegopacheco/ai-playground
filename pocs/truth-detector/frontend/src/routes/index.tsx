import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { useAnalyze, useAnalysisQuery, useWeeklyScores } from "../hooks/useAnalysis";
import { useAnalysisSSE } from "../hooks/useAnalysisSSE";
import UserInput from "../components/UserInput";
import AnalysisProgress from "../components/AnalysisProgress";
import ScoreSummary from "../components/ScoreSummary";
import CommitCard from "../components/CommitCard";
import WeeklyChart from "../components/WeeklyChart";

export const Route = createFileRoute("/")({
  component: AnalyzePage,
});

function parseUsername(input: string): string {
  const trimmed = input.trim();
  try {
    const url = new URL(trimmed);
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length > 0) return parts[0];
  } catch {
    /* not a URL */
  }
  return trimmed;
}

function AnalyzePage() {
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [username, setUsername] = useState<string | null>(null);
  const analyzeMutation = useAnalyze();
  const sseState = useAnalysisSSE(analysisId);
  const analysisQuery = useAnalysisQuery(
    sseState.phase === "complete" ? analysisId : null
  );
  const weeklyQuery = useWeeklyScores(
    sseState.phase === "complete" ? username : null
  );

  const handleAnalyze = (input: string) => {
    const user = parseUsername(input);
    setUsername(user);
    setAnalysisId(null);
    analyzeMutation.mutate(user, {
      onSuccess: (data) => {
        setAnalysisId(data.id);
      },
    });
  };

  const analysis = analysisQuery.data;
  const weeklyScores = weeklyQuery.data;

  return (
    <div className="space-y-6">
      <UserInput
        onAnalyze={handleAnalyze}
        isLoading={analyzeMutation.isPending}
      />

      {analysisId && sseState.phase !== "idle" && (
        <AnalysisProgress phase={sseState.phase} error={sseState.error} />
      )}

      {analysis && sseState.phase === "complete" && (
        <>
          <ScoreSummary analysis={analysis} />

          <div>
            <h2 className="text-lg font-semibold text-gray-200 mb-3">
              Commit Results
            </h2>
            <div className="space-y-3">
              {analysis.commits.map((commit) => (
                <CommitCard key={commit.id} commit={commit} />
              ))}
            </div>
          </div>

          {weeklyScores && weeklyScores.length > 0 && (
            <WeeklyChart scores={weeklyScores} />
          )}
        </>
      )}
    </div>
  );
}
