interface GameOverProps {
  score: number
  moves: number
  onPlayAgain: () => void
}

export default function GameOver({ score, moves, onPlayAgain }: GameOverProps) {
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-2xl p-8 max-w-sm w-full mx-4 text-center shadow-2xl border border-gray-700">
        <h2 className="text-3xl font-bold text-green-400 mb-6">You Won!</h2>
        <div className="space-y-3 mb-8">
          <div className="flex justify-between text-gray-300">
            <span>Score</span>
            <span className="font-bold text-white">{score}</span>
          </div>
          <div className="flex justify-between text-gray-300">
            <span>Moves</span>
            <span className="font-bold text-white">{moves}</span>
          </div>
        </div>
        <button
          onClick={onPlayAgain}
          className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-xl transition-colors"
        >
          Play Again
        </button>
      </div>
    </div>
  )
}
