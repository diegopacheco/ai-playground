import { ProgressState } from '../types'

interface ProgressBarProps {
  progress: ProgressState
}

function ProgressBar({ progress }: ProgressBarProps) {
  const percentage = progress.totalSteps > 0
    ? Math.round((progress.currentStep / progress.totalSteps) * 100)
    : 0

  return (
    <div className="bg-gray-800 rounded-lg p-6 mb-6">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm text-gray-300">Progress</span>
        <span className="text-sm text-blue-400">{percentage}%</span>
      </div>
      <div className="h-3 bg-gray-700 rounded-full overflow-hidden mb-3">
        <div
          className="h-full bg-blue-500 transition-all duration-300 ease-out"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="flex items-center gap-2">
        {!progress.isComplete && (
          <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        )}
        <span className="text-gray-300">{progress.message}</span>
      </div>
      <div className="mt-2 text-xs text-gray-500">
        Step {progress.currentStep} of {progress.totalSteps}
      </div>
    </div>
  )
}

export default ProgressBar
