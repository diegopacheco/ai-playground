import { AnalyzeResponse, ProgressState } from '../types'
import ScoreCard from '../components/ScoreCard'
import ModelResult from '../components/ModelResult'
import ProgressBar from '../components/ProgressBar'

interface ScorePageProps {
  result: AnalyzeResponse | null
  loading: boolean
  progress: ProgressState
}

function ScorePage({ result, loading, progress }: ScorePageProps) {
  if (!result && !loading) {
    return (
      <div className="text-center text-gray-400 py-12">
        <p className="text-lg">No analysis results yet.</p>
        <p className="mt-2">Go to the Prompt tab and enter a prompt to analyze.</p>
      </div>
    )
  }

  const dimensions = result?.scores ? [
    { label: 'Quality', value: result.scores.quality },
    { label: 'Stack Definitions', value: result.scores.stack_definitions },
    { label: 'Clear Goals', value: result.scores.clear_goals },
    { label: 'Non-obvious Decisions', value: result.scores.non_obvious_decisions },
    { label: 'Security & Operations', value: result.scores.security_operations },
    { label: 'Overall Effectiveness', value: result.scores.overall_effectiveness },
  ] : []

  return (
    <div className="space-y-8">
      {loading && <ProgressBar progress={progress} />}

      {result?.scores && (
        <section>
          <h2 className="text-xl font-semibold mb-4">Dimension Scores</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {dimensions.map((dim) => (
              <ScoreCard key={dim.label} label={dim.label} score={dim.value} />
            ))}
          </div>
        </section>
      )}

      {result?.model_results && result.model_results.length > 0 && (
        <section>
          <h2 className="text-xl font-semibold mb-4">Model Analysis</h2>
          <div className="space-y-4">
            {result.model_results.map((model) => (
              <ModelResult key={model.model} result={model} />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

export default ScorePage
