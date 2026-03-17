import { useState, useEffect } from 'react'
import { QueryRecord } from '../types'
import { getQueries } from '../api/queries'

function History() {
  const [queries, setQueries] = useState<QueryRecord[]>([])

  useEffect(() => {
    getQueries().then(setQueries)
  }, [])

  const statusColor = (s: string) => {
    switch (s) {
      case 'success': return 'text-green-400 bg-green-400/10'
      case 'failed': return 'text-red-400 bg-red-400/10'
      case 'blocked': return 'text-yellow-400 bg-yellow-400/10'
      default: return 'text-slate-400 bg-slate-400/10'
    }
  }

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white">Query History</h2>
      {queries.length === 0 && (
        <p className="text-slate-400">No queries yet.</p>
      )}
      {queries.map(q => (
        <div key={q.id} className="bg-slate-800 rounded-lg border border-slate-700 p-4">
          <div className="flex items-center justify-between mb-2">
            <p className="text-white font-medium">{q.question}</p>
            <span className={`px-2 py-0.5 rounded text-xs font-semibold ${statusColor(q.status)}`}>
              {q.status}
            </span>
          </div>
          {q.generated_sql && (
            <pre className="bg-slate-900 rounded p-2 text-xs text-green-400 overflow-x-auto mb-2">{q.generated_sql}</pre>
          )}
          <p className="text-slate-500 text-xs">{q.created_at}</p>
        </div>
      ))}
    </div>
  )
}

export default History
