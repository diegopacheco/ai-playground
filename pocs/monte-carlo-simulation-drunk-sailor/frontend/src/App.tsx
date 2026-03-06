import { useState, useRef, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import './App.css'

interface SimulationResult {
  fell_in_water: number
  reached_ship: number
  total: number
  fall_percentage: number
  ship_percentage: number
  paths: number[][]
}

async function runSimulation(params: { num_simulations: number; pier_length: number }): Promise<SimulationResult> {
  const res = await fetch('http://localhost:8080/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  return res.json()
}

function drawPaths(canvas: HTMLCanvasElement, paths: number[][], pierLength: number) {
  const ctx = canvas.getContext('2d')!
  const width = canvas.width
  const height = canvas.height
  ctx.clearRect(0, 0, width, height)

  ctx.fillStyle = '#0a0a23'
  ctx.fillRect(0, 0, width, height)

  ctx.strokeStyle = '#1a2744'
  ctx.lineWidth = 1
  const yScale = height / pierLength
  for (let i = 0; i <= pierLength; i++) {
    const y = height - i * yScale
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(width, y)
    ctx.stroke()
  }

  ctx.fillStyle = '#3498db33'
  ctx.fillRect(0, height - yScale, width, yScale)

  ctx.fillStyle = '#2ecc7133'
  ctx.fillRect(0, 0, width, yScale)

  ctx.font = '11px system-ui'
  ctx.fillStyle = '#3498db'
  ctx.fillText('WATER', 5, height - 5)
  ctx.fillStyle = '#2ecc71'
  ctx.fillText('SHIP', 5, 14)

  let maxSteps = 0
  for (const path of paths) {
    if (path.length > maxSteps) maxSteps = path.length
  }
  const xScale = width / Math.max(maxSteps - 1, 1)

  for (const path of paths) {
    const fellInWater = path[path.length - 1] <= 0
    ctx.strokeStyle = fellInWater ? '#3498db55' : '#2ecc7155'
    ctx.lineWidth = 1.2
    ctx.beginPath()
    for (let i = 0; i < path.length; i++) {
      const x = i * xScale
      const y = height - path[i] * yScale
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()
  }
}

function App() {
  const [numSimulations, setNumSimulations] = useState(1000)
  const [pierLength, setPierLength] = useState(20)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const mutation = useMutation({
    mutationFn: runSimulation,
  })

  const result = mutation.data

  useEffect(() => {
    if (result && canvasRef.current) {
      drawPaths(canvasRef.current, result.paths, pierLength)
    }
  }, [result, pierLength])

  return (
    <>
      <h1>Monte Carlo: Drunk Sailor</h1>
      <p className="subtitle">
        A sailor leaves a bar on a pier, taking random steps toward the water or toward his ship.
        Run thousands of simulations to see the pattern emerge from the noise.
      </p>

      <div className="controls">
        <div className="control-group">
          <label>Simulations</label>
          <input
            type="number"
            value={numSimulations}
            onChange={(e) => setNumSimulations(Number(e.target.value))}
            min={1}
            max={100000}
          />
        </div>
        <div className="control-group">
          <label>Pier Length</label>
          <input
            type="number"
            value={pierLength}
            onChange={(e) => setPierLength(Number(e.target.value))}
            min={4}
            max={200}
          />
        </div>
        <button
          onClick={() => mutation.mutate({ num_simulations: numSimulations, pier_length: pierLength })}
          disabled={mutation.isPending}
        >
          {mutation.isPending ? 'Simulating...' : 'Run Simulation'}
        </button>
      </div>

      {mutation.isPending && <p className="loading">Running simulation...</p>}

      {result && (
        <div className="results">
          <div className="stats">
            <h2>Results</h2>
            <div className="stat-row">
              <span className="stat-label">Total Simulations</span>
              <span className="stat-value">{result.total.toLocaleString()}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Fell in Water</span>
              <span className="stat-value water">
                {result.fell_in_water.toLocaleString()} ({result.fall_percentage.toFixed(1)}%)
              </span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Reached Ship</span>
              <span className="stat-value ship">
                {result.reached_ship.toLocaleString()} ({result.ship_percentage.toFixed(1)}%)
              </span>
            </div>
            <div className="bar-container">
              <div className="bar-water" style={{ width: `${result.fall_percentage}%` }}>
                {result.fall_percentage > 10 ? `${result.fall_percentage.toFixed(1)}%` : ''}
              </div>
              <div className="bar-ship" style={{ width: `${result.ship_percentage}%` }}>
                {result.ship_percentage > 10 ? `${result.ship_percentage.toFixed(1)}%` : ''}
              </div>
            </div>
            <div className="stat-row" style={{ marginTop: '1rem', borderBottom: 'none' }}>
              <span className="stat-label">Sailor starts at position {Math.floor(pierLength / 2)} (middle of pier)</span>
            </div>
          </div>
          <div className="paths-container">
            <h2>Sample Paths (up to 50)</h2>
            <canvas ref={canvasRef} width={500} height={350} />
          </div>
        </div>
      )}
    </>
  )
}

export default App
