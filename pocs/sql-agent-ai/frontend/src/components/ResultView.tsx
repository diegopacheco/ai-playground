import { QueryResultEvent } from '../types'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts'

interface Props {
  result: QueryResultEvent
}

const COLORS = ['#06b6d4', '#8b5cf6', '#f59e0b', '#10b981', '#ef4444', '#3b82f6', '#ec4899', '#14b8a6', '#f97316', '#6366f1']

function ResultView({ result }: Props) {
  const { columns, rows, sql } = result

  if (rows.length === 0) {
    return (
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
        <p className="text-slate-400">No results found.</p>
      </div>
    )
  }

  const chartData = rows.map(row => {
    const obj: Record<string, any> = {}
    columns.forEach((col, i) => {
      obj[col] = row[i]
    })
    return obj
  })

  const hasNumericColumn = columns.some((_, i) => rows.some(r => typeof r[i] === 'number'))
  const labelCol = columns[0]
  const numericCols = columns.filter((_, i) => rows.some(r => typeof r[i] === 'number'))
  const showPie = rows.length <= 8 && numericCols.length === 1
  const showLine = rows.length > 10 || columns.some(c => c.toLowerCase().includes('date') || c.toLowerCase().includes('month') || c.toLowerCase().includes('forecast'))

  return (
    <div className="space-y-6">
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
        <h3 className="text-sm font-semibold text-slate-400 mb-2">Generated SQL</h3>
        <pre className="bg-slate-900 rounded p-3 text-sm text-green-400 overflow-x-auto">{sql}</pre>
      </div>

      {hasNumericColumn && (
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
          <h3 className="text-sm font-semibold text-slate-400 mb-4">Chart</h3>
          <div className="h-80">
            {showPie ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    dataKey={numericCols[0]}
                    nameKey={labelCol}
                    cx="50%"
                    cy="50%"
                    outerRadius={120}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {chartData.map((_, idx) => (
                      <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', color: '#e2e8f0' }} />
                </PieChart>
              </ResponsiveContainer>
            ) : showLine ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey={labelCol} stroke="#94a3b8" tick={{ fontSize: 12 }} angle={-30} textAnchor="end" height={60} />
                  <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', color: '#e2e8f0' }} />
                  {numericCols.map((col, idx) => (
                    <Line key={col} type="monotone" dataKey={col} stroke={COLORS[idx % COLORS.length]} strokeWidth={2} dot={{ fill: COLORS[idx % COLORS.length] }} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey={labelCol} stroke="#94a3b8" tick={{ fontSize: 12 }} angle={-30} textAnchor="end" height={60} />
                  <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', color: '#e2e8f0' }} />
                  {numericCols.map((col, idx) => (
                    <Bar key={col} dataKey={col} fill={COLORS[idx % COLORS.length]} radius={[4, 4, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      )}

      <div className="bg-slate-800 rounded-lg border border-slate-700 p-4 overflow-x-auto">
        <h3 className="text-sm font-semibold text-slate-400 mb-3">Data ({rows.length} rows)</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              {columns.map(col => (
                <th key={col} className="text-left py-2 px-3 text-cyan-400 font-semibold">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                {row.map((val, j) => (
                  <td key={j} className="py-2 px-3 text-slate-300">
                    {val === null ? '-' : String(val)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default ResultView
