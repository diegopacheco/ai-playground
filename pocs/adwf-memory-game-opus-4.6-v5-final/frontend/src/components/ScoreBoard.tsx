import { useState, useEffect } from "react"

interface ScoreBoardProps {
  moves: number
  matchesFound: number
  totalPairs: number
  isComplete: boolean
}

export default function ScoreBoard({ moves, matchesFound, totalPairs, isComplete }: ScoreBoardProps) {
  const [elapsed, setElapsed] = useState(0)
  const [startTime] = useState(() => Date.now())

  useEffect(() => {
    if (isComplete) return
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000))
    }, 1000)
    return () => clearInterval(interval)
  }, [startTime, isComplete])

  const minutes = Math.floor(elapsed / 60)
  const seconds = elapsed % 60

  return (
    <div className="flex gap-6 justify-center items-center bg-gray-800 rounded-xl px-6 py-3 text-white shadow-md" data-testid="score-board">
      <div className="text-center">
        <div className="text-xs uppercase tracking-wide text-gray-400">Moves</div>
        <div className="text-2xl font-bold">{moves}</div>
      </div>
      <div className="w-px h-10 bg-gray-600" />
      <div className="text-center">
        <div className="text-xs uppercase tracking-wide text-gray-400">Matches</div>
        <div className="text-2xl font-bold">
          {matchesFound}/{totalPairs}
        </div>
      </div>
      <div className="w-px h-10 bg-gray-600" />
      <div className="text-center">
        <div className="text-xs uppercase tracking-wide text-gray-400">Time</div>
        <div className="text-2xl font-bold">
          {minutes}:{seconds.toString().padStart(2, "0")}
        </div>
      </div>
    </div>
  )
}
