interface ScoreCardProps {
  label: string
  score: number
}

function ScoreCard({ label, score }: ScoreCardProps) {
  const getScoreColor = (s: number) => {
    if (s >= 4) return 'text-green-400'
    if (s >= 3) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getBarWidth = (s: number) => `${(s / 5) * 100}%`

  const getBarColor = (s: number) => {
    if (s >= 4) return 'bg-green-500'
    if (s >= 3) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm text-gray-300">{label}</span>
        <span className={`text-2xl font-bold ${getScoreColor(score)}`}>{score}/5</span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${getBarColor(score)} transition-all duration-300`}
          style={{ width: getBarWidth(score) }}
        />
      </div>
    </div>
  )
}

export default ScoreCard
