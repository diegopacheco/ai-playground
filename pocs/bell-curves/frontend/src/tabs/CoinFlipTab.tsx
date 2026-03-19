import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

interface ChartData {
  heads: number
  count: number
}

function CoinFlipTab() {
  const [data, setData] = useState<ChartData[]>([])
  const [trials, setTrials] = useState(2000)
  const [flips, setFlips] = useState(100)
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState({ mean: 0, stddev: 0 })

  const run = async () => {
    setLoading(true)
    const res = await fetch(
      `http://localhost:8080/api/coin-flip?trials=${trials}&flips=${flips}`
    )
    const json = await res.json()
    const buckets: number[] = json.buckets

    const min = Math.max(0, Math.floor(flips * 0.3))
    const max = Math.min(buckets.length - 1, Math.ceil(flips * 0.7))
    const chartData: ChartData[] = []
    let totalHeads = 0
    let totalCount = 0

    for (let i = min; i <= max; i++) {
      if (buckets[i] > 0) {
        chartData.push({ heads: i, count: buckets[i] })
      }
      totalHeads += i * buckets[i]
      totalCount += buckets[i]
    }

    const mean = totalHeads / totalCount
    let variance = 0
    for (let i = 0; i < buckets.length; i++) {
      variance += buckets[i] * Math.pow(i - mean, 2)
    }
    variance /= totalCount

    setStats({ mean: Math.round(mean * 100) / 100, stddev: Math.round(Math.sqrt(variance) * 100) / 100 })
    setData(chartData)
    setLoading(false)
  }

  return (
    <div className="tab-content">
      <h2>Coin Flip Distribution</h2>
      <p className="description">
        Flip a fair coin many times per trial and count the number of heads.
        As described in the Quanta Magazine article, De Moivre showed that flipping
        a coin 100 times yields between 45 and 55 heads about 68% of the time -
        forming a perfect bell curve around the expected value of 50.
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
          Flips per trial:
          <input
            type="number"
            value={flips}
            onChange={e => setFlips(Number(e.target.value))}
            min={10}
            max={500}
            step={10}
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
                  dataKey="heads"
                  stroke="#666"
                  label={{ value: 'Number of Heads', position: 'bottom', fill: '#666', offset: 0 }}
                  tick={{ fontSize: 11 }}
                />
                <YAxis
                  stroke="#666"
                  label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: '#666' }}
                />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
                  labelStyle={{ color: '#aaa' }}
                />
                <Bar dataKey="count" fill="#667eea" radius={[2, 2, 0, 0]} />
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
              <div className="value">{flips / 2}</div>
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

export default CoinFlipTab
