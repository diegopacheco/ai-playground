import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

function fmt(n) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(n)
}

export default function ResultsPage({ results, inputData, onBack }) {
  const r = results

  return (
    <div>
      <div className="results-header">
        <h2 style={{ fontSize: '1.5rem' }}>Your Retirement Plan</h2>
        <button className="btn-secondary" onClick={onBack}>Back to Simulation</button>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">Total Savings at Retirement</div>
          <div className="value">{fmt(r.totalSavingsAtRetirement)}</div>
        </div>
        <div className="stat-card">
          <div className="label">Total Needed</div>
          <div className="value">{fmt(r.totalNeededForRetirement)}</div>
        </div>
        <div className="stat-card">
          <div className="label">Monthly Income</div>
          <div className="value">{fmt(r.monthlyIncomeFromSavings)}</div>
        </div>
        <div className={`stat-card ${r.onTrack ? 'on-track' : 'off-track'}`}>
          <div className="label">Status</div>
          <div className="value">{r.onTrack ? 'ON TRACK' : 'OFF TRACK'}</div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <h3 className="section-title">Key Metrics</h3>
        <div className="metrics-grid">
          <div className="metric-item">
            <span className="metric-label">Years to Retirement</span>
            <span className="metric-value">{r.yearsToRetirement}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Years in Retirement</span>
            <span className="metric-value">{r.yearsInRetirement}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Total Contributions</span>
            <span className="metric-value">{fmt(r.totalContributions)}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Interest Earned</span>
            <span className="metric-value">{fmt(r.totalInterestEarned)}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Inflation-Adjusted Monthly Need</span>
            <span className="metric-value">{fmt(r.inflationAdjustedMonthlyIncome)}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Risk Level</span>
            <span className={`metric-value risk-${r.riskAssessment}`}>{r.riskAssessment}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Savings Gap</span>
            <span className="metric-value">{fmt(r.savingsGap)}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Required Monthly Savings</span>
            <span className="metric-value">{fmt(r.requiredMonthlySavings)}</span>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <h3 className="section-title">Balance Growth</h3>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={r.yearlyProjections}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="age" />
              <YAxis tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
              <Tooltip formatter={(v) => fmt(v)} />
              <Area type="monotone" dataKey="endBalance" stroke="#059669" fill="#ecfdf5" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <h3 className="section-title">Recommendations</h3>
        <ul className="recommendations-list">
          {r.recommendations.map((rec, i) => (
            <li key={i}>{rec}</li>
          ))}
        </ul>
      </div>

      <div className="card">
        <h3 className="section-title">Year-by-Year Projections</h3>
        <div style={{ overflowX: 'auto' }}>
          <table className="projection-table">
            <thead>
              <tr>
                <th>Age</th>
                <th>Start Balance</th>
                <th>Contributions</th>
                <th>Interest Earned</th>
                <th>End Balance</th>
              </tr>
            </thead>
            <tbody>
              {r.yearlyProjections.map((p) => (
                <tr key={p.year}>
                  <td>{p.age}</td>
                  <td>{fmt(p.startBalance)}</td>
                  <td>{fmt(p.contributions)}</td>
                  <td>{fmt(p.interestEarned)}</td>
                  <td>{fmt(p.endBalance)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
