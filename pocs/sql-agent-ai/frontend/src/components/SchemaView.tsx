import { useState, useEffect } from 'react'
import { TableInfo } from '../types'
import { getSchema } from '../api/queries'

function SchemaView() {
  const [tables, setTables] = useState<TableInfo[]>([])

  useEffect(() => {
    getSchema().then(setTables)
  }, [])

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white">Database Schema</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {tables.map(t => (
          <div key={t.name} className="bg-slate-800 rounded-lg border border-slate-700 p-4">
            <h3 className="text-cyan-400 font-semibold mb-3">{t.name}</h3>
            <div className="space-y-1">
              {t.columns.map(c => (
                <div key={c.name} className="flex justify-between text-sm">
                  <span className="text-white">{c.name}</span>
                  <span className="text-slate-500">{c.data_type}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default SchemaView
