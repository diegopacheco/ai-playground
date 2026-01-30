interface LoadingViewProps {
  progress: {
    step: number
    message: string
    total: number
  }
}

function LoadingView({ progress }: LoadingViewProps) {
  const percentage = progress.total > 0 ? (progress.step / progress.total) * 100 : 0

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800 flex items-center justify-center p-6">
      <div className="max-w-md w-full bg-white/10 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-white/20 text-center">
        <div className="mb-8">
          <div className="w-24 h-24 mx-auto mb-6 relative">
            <div className="absolute inset-0 border-4 border-purple-500/30 rounded-full" />
            <div
              className="absolute inset-0 border-4 border-purple-500 rounded-full border-t-transparent animate-spin"
              style={{ animationDuration: '1s' }}
            />
            <div className="absolute inset-4 bg-purple-500/20 rounded-full flex items-center justify-center">
              <span className="text-white text-2xl font-bold">{progress.step}</span>
            </div>
          </div>

          <h2 className="text-2xl font-bold text-white mb-2">Generating Your Training</h2>
          <p className="text-purple-200">{progress.message}</p>
        </div>

        <div className="w-full bg-white/20 rounded-full h-3 mb-4">
          <div
            className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full transition-all duration-500"
            style={{ width: `${percentage}%` }}
          />
        </div>

        <p className="text-purple-300 text-sm">
          Step {progress.step} of {progress.total}
        </p>

        <div className="mt-8 space-y-3">
          <Step number={1} label="Generating training content" active={progress.step === 1} done={progress.step > 1} />
          <Step number={2} label="Creating quiz questions" active={progress.step === 2} done={progress.step > 2} />
          <Step number={3} label="Finalizing your training" active={progress.step === 3} done={progress.step > 3} />
        </div>
      </div>
    </div>
  )
}

function Step({ number, label, active, done }: { number: number; label: string; active: boolean; done: boolean }) {
  return (
    <div
      className={`flex items-center gap-3 p-3 rounded-lg transition-all ${
        active ? 'bg-purple-500/30' : done ? 'bg-green-500/20' : 'bg-white/5'
      }`}
    >
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
          done ? 'bg-green-500 text-white' : active ? 'bg-purple-500 text-white' : 'bg-white/20 text-purple-300'
        }`}
      >
        {done ? '✓' : number}
      </div>
      <span className={`${active || done ? 'text-white' : 'text-purple-300'}`}>{label}</span>
    </div>
  )
}

export default LoadingView
