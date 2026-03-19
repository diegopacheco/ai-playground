import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

interface ChartData {
  bucket: string
  count: number
}

function DiceRollTab() {
  const [data, setData] = useState<ChartData[]>([])
  const [trials, setTrials] = useState(3000)
  const [rolls, setRolls] = useState(50)
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState({ mean: 0, stddev: 0 })

  const run = async () => {
    setLoading(true)
    const res = await fetch(
      `http://localhost:8080/api/dice-roll?trials=${trials}&rolls=${rolls}`
    )
    const json = await res.json()
    const averages: number[] = json.averages

    const numBuckets = 40
    const min = Math.min(...averages)
    const max = Math.max(...averages)
    const step = (max - min) / numBuckets
    const buckets = new Array(numBuckets).fill(0)

    let sum = 0
    for (const avg of averages) {
      sum += avg
      const idx = Math.min(Math.floor((avg - min) / step), numBuckets - 1)
      buckets[idx]++
    }

    const mean = sum / averages.length
    let variance = 0
    for (const avg of averages) {
      variance += Math.pow(avg - mean, 2)
    }
    variance /= averages.length

    setStats({
      mean: Math.round(mean * 1000) / 1000,
      stddev: Math.round(Math.sqrt(variance) * 1000) / 1000,
    })

    const chartData: ChartData[] = buckets.map((count, i) => ({
      bucket: (min + (i + 0.5) * step).toFixed(2),
      count,
    }))

    setData(chartData)
    setLoading(false)
  }

  return (
    <div className="tab-content">
      <h2>Dice Roll Averages</h2>
      <p className="description">
        Roll a die many times and compute the average. Repeat this process thousands
        of times. As the Quanta article explains, averaging multiple dice rolls
        produces values clustered around 3.5 in a bell curve shape. The more rolls
        per trial, the tighter the bell curve becomes.
      </p>
      <div className="controls">
        <label>
          Trials:
          <input
            type="number"
            value={trials}
            onChange={e => setTrials(Number(e.target.value))}
            min={100}
            max={10000}
            step={100}
          />
        </label>
        <label>
          Rolls per trial:
          <input
            type="number"
            value={rolls}
            onChange={e => setRolls(Number(e.target.value))}
            min={5}
            max={500}
            step={5}
          />
        </label>
        <button className="run-btn" onClick={run} disabled={loading}>
          {loading ? 'Running...' : 'Run Simulation'}
        </button>
      </div>
      {data.length > 0 && (
        <>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis
                  dataKey="bucket"
                  stroke="#666"
                  label={{ value: 'Average Dice Value', position: 'bottom', fill: '#666', offset: 0 }}
                  tick={{ fontSize: 10 }}
                  interval={4}
                />
                <YAxis
                  stroke="#666"
                  label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: '#666' }}
                />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
                  labelStyle={{ color: '#aaa' }}
                />
                <Bar dataKey="count" fill="#a855f7" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="stats">
            <div className="stat-card">
              <div className="label">Mean</div>
              <div className="value">{stats.mean}</div>
            </div>
            <div className="stat-card">
              <div className="label">Std Deviation</div>
              <div className="value">{stats.stddev}</div>
            </div>
            <div className="stat-card">
              <div className="label">Expected Mean</div>
              <div className="value">3.500</div>
            </div>
            <div className="stat-card">
              <div className="label">Trials</div>
              <div className="value">{trials}</div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default DiceRollTab
