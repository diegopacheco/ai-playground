import { useState } from 'react'

interface Props {
  onSubmit: (question: string) => void
  isRunning: boolean
  onNewQuery: () => void
}

const SUGGESTIONS = [
  'Give me the top 10 salesman report',
  'Give the top 5 products in sales',
  'Give me the worst 10 products',
  'Tell me which state buys more and less top 5',
  'Tell me sales forecast for next month',
]

function QueryInput({ onSubmit, isRunning, onNewQuery }: Props) {
  const [input, setInput] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isRunning) {
      onSubmit(input.trim())
    }
  }

  const handleSuggestion = (s: string) => {
    onNewQuery()
    setInput(s)
    onSubmit(s)
  }

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="flex gap-3">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask a question about sales data..."
          className="flex-1 bg-slate-800 border border-slate-600 rounded-lg px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          disabled={isRunning}
        />
        <button
          type="submit"
          disabled={isRunning || !input.trim()}
          className="bg-cyan-600 hover:bg-cyan-700 disabled:bg-slate-700 disabled:text-slate-500 text-white px-6 py-3 rounded-lg font-semibold transition"
        >
          {isRunning ? 'Running...' : 'Ask'}
        </button>
        <button
          type="button"
          onClick={onNewQuery}
          className="bg-slate-700 hover:bg-slate-600 text-slate-300 px-4 py-3 rounded-lg transition"
        >
          Clear
        </button>
      </form>

      <div className="flex flex-wrap gap-2">
        {SUGGESTIONS.map(s => (
          <button
            key={s}
            onClick={() => handleSuggestion(s)}
            disabled={isRunning}
            className="bg-slate-800 border border-slate-700 hover:border-cyan-500 text-slate-400 hover:text-cyan-400 px-3 py-1.5 rounded-full text-sm transition disabled:opacity-50"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  )
}

export default QueryInput
