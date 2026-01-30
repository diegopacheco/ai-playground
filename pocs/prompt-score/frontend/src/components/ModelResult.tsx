import { ModelResult as ModelResultType } from '../types'

interface ModelResultProps {
  result: ModelResultType
}

function ModelResult({ result }: ModelResultProps) {
  const getScoreColor = (s: number | null) => {
    if (s === null) return 'text-gray-400'
    if (s >= 4) return 'text-green-400'
    if (s >= 3) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex justify-between items-start mb-3">
        <h3 className="text-lg font-medium text-blue-400">{result.model}</h3>
        <span className={`text-xl font-bold ${getScoreColor(result.score)}`}>
          {result.score !== null ? `${result.score}/5` : 'N/A'}
        </span>
      </div>
      <div className="bg-gray-700 rounded p-3">
        <p className="text-gray-300 text-sm whitespace-pre-wrap">{result.recommendations}</p>
      </div>
    </div>
  )
}

export default ModelResult
