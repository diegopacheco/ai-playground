import type { SSEPhase } from "../hooks/useAnalysisSSE";

interface AnalysisProgressProps {
  phase: SSEPhase;
  error: string | null;
}

function Spinner() {
  return (
    <span className="inline-block w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
  );
}

function AnalysisProgress({ phase, error }: AnalysisProgressProps) {
  if (phase === "error") {
    return (
      <div className="bg-red-50 border border-red-300 rounded-lg p-4 text-red-600">
        Error: {error}
      </div>
    );
  }

  const phases = [
    { key: "started", label: "Starting analysis..." },
    { key: "commits_fetched", label: "Fetching commits..." },
    { key: "llm_thinking", label: "Analyzing with LLM..." },
    { key: "analyzed", label: "Processing results..." },
    { key: "complete", label: "Complete!" },
  ];

  const currentIndex = phases.findIndex((p) => p.key === phase);
  const progress = phase === "complete" ? 100 : ((currentIndex + 1) / phases.length) * 100;

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 space-y-3">
      <div className="flex items-center gap-2">
        {phase !== "complete" ? (
          <Spinner />
        ) : (
          <span className="text-green-600 text-lg">&#10003;</span>
        )}
        <span className="text-gray-800 font-medium">
          {phases.find((p) => p.key === phase)?.label ?? "Waiting..."}
        </span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className="bg-blue-500 h-2 rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}

export default AnalysisProgress;
