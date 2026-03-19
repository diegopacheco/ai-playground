import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

interface ChartData {
  bucket: string
  count: number
}

function CLTTab() {
  const [data, setData] = useState<ChartData[]>([])
  const [numSamples, setNumSamples] = useState(3000)
  const [sampleSize, setSampleSize] = useState(30)
  const [source, setSource] = useState('uniform')
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState({ mean: 0, stddev: 0 })

  const run = async () => {
    setLoading(true)
    const res = await fetch(
      `http://localhost:8080/api/clt?numSamples=${numSamples}&sampleSize=${sampleSize}&source=${source}`
    )
    const json = await res.json()
    const means: number[] = json.sampleMeans

    const numBuckets = 45
    const min = Math.min(...means)
    const max = Math.max(...means)
    const step = (max - min) / numBuckets
    const buckets = new Array(numBuckets).fill(0)

    let sum = 0
    for (const m of means) {
      sum += m
      const idx = Math.min(Math.floor((m - min) / step), numBuckets - 1)
      buckets[idx]++
    }

    const mean = sum / means.length
    let variance = 0
    for (const m of means) {
      variance += Math.pow(m - mean, 2)
    }
    variance /= means.length

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

  const sourceDescriptions: Record<string, string> = {
    uniform: 'Uniform distribution (flat, 0-10) - every value equally likely',
    exponential: 'Exponential distribution (heavily right-skewed) - most values small, few very large',
    bimodal: 'Bimodal distribution (two peaks at 3 and 8) - two distinct clusters of values',
  }

  return (
    <div className="tab-content">
      <h2>Central Limit Theorem</h2>
      <p className="description">
        The Central Limit Theorem is the core insight from the Quanta article:
        no matter what the original distribution looks like, the distribution of
        sample means always converges to a bell curve. Try different source
        distributions below - uniform, exponential, or bimodal - and watch
        how the sample means always form a Gaussian shape. Laplace proved this
        foundational result that explains why bell curves appear everywhere in nature.
      </p>
      <div className="controls">
        <label>
          Source:
          <select value={source} onChange={e => setSource(e.target.value)}>
            <option value="uniform">Uniform</option>
            <option value="exponential">Exponential</option>
            <option value="bimodal">Bimodal</option>
          </select>
        </label>
        <label>
          Samples:
          <input
            type="number"
            value={numSamples}
            onChange={e => setNumSamples(Number(e.target.value))}
            min={100}
            max={10000}
            step={100}
          />
        </label>
        <label>
          Sample size:
          <input
            type="number"
            value={sampleSize}
            onChange={e => setSampleSize(Number(e.target.value))}
            min={2}
            max={200}
            step={1}
          />
        </label>
        <button className="run-btn" onClick={run} disabled={loading}>
          {loading ? 'Running...' : 'Run Simulation'}
        </button>
      </div>
      <p className="description" style={{ marginBottom: 16, color: '#9977cc' }}>
        {sourceDescriptions[source]}
      </p>
      {data.length > 0 && (
        <>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis
                  dataKey="bucket"
                  stroke="#666"
                  label={{ value: 'Sample Mean', position: 'bottom', fill: '#666', offset: 0 }}
                  tick={{ fontSize: 10 }}
                  interval={5}
                />
                <YAxis
                  stroke="#666"
                  label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: '#666' }}
                />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
                  labelStyle={{ color: '#aaa' }}
                />
                <Bar dataKey="count" fill="#06b6d4" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="stats">
            <div className="stat-card">
              <div className="label">Mean of Means</div>
              <div className="value">{stats.mean}</div>
            </div>
            <div className="stat-card">
              <div className="label">Std Deviation</div>
              <div className="value">{stats.stddev}</div>
            </div>
            <div className="stat-card">
              <div className="label">Source</div>
              <div className="value" style={{ fontSize: '1rem' }}>{source}</div>
            </div>
            <div className="stat-card">
              <div className="label">Sample Size</div>
              <div className="value">{sampleSize}</div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default CLTTab
