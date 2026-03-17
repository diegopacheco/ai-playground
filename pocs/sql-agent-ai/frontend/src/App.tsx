import { useState } from 'react'
import QueryInput from './components/QueryInput'
import ResultView from './components/ResultView'
import EventLog from './components/EventLog'
import History from './components/History'
import SchemaView from './components/SchemaView'
import { useQuerySSE } from './hooks/useQuerySSE'
import { createQuery } from './api/queries'

function App() {
  const [queryId, setQueryId] = useState<string | null>(null)
  const [question, setQuestion] = useState('')
  const [tab, setTab] = useState<'query' | 'history' | 'schema'>('query')
  const { events, result, error, isRunning, reset } = useQuerySSE(queryId)

  const handleSubmit = async (q: string) => {
    reset()
    setQuestion(q)
    const res = await createQuery({ question: q })
    setQueryId(res.id)
  }

  const handleNewQuery = () => {
    setQueryId(null)
    setQuestion('')
    reset()
  }

  return (
    <div className="min-h-screen bg-slate-900">
      <header className="bg-slate-800 border-b border-slate-700 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <h1 className="text-2xl font-bold text-cyan-400">SQL Agent</h1>
          <nav className="flex gap-2">
            <button
              onClick={() => setTab('query')}
              className={`px-4 py-2 rounded ${tab === 'query' ? 'bg-cyan-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}
            >
              Query
            </button>
            <button
              onClick={() => setTab('history')}
              className={`px-4 py-2 rounded ${tab === 'history' ? 'bg-cyan-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}
            >
              History
            </button>
            <button
              onClick={() => setTab('schema')}
              className={`px-4 py-2 rounded ${tab === 'schema' ? 'bg-cyan-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}
            >
              Schema
            </button>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-6">
        {tab === 'query' && (
          <div className="space-y-6">
            <QueryInput onSubmit={handleSubmit} isRunning={isRunning} onNewQuery={handleNewQuery} />

            {question && (
              <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <p className="text-slate-400 text-sm">Question:</p>
                <p className="text-white text-lg">{question}</p>
              </div>
            )}

            {events.length > 0 && <EventLog events={events} />}

            {error && (
              <div className="bg-red-900/30 border border-red-700 rounded-lg p-4">
                <p className="text-red-400 font-semibold">Error</p>
                <p className="text-red-300">{error}</p>
              </div>
            )}

            {result && <ResultView result={result} />}
          </div>
        )}

        {tab === 'history' && <History />}
        {tab === 'schema' && <SchemaView />}
      </main>
    </div>
  )
}

export default App
