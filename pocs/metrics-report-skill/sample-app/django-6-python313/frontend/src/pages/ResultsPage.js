import React from 'react';
import { Link } from 'react-router-dom';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

function formatCurrency(value) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value);
}

function ResultsPage({ results, inputData }) {
  if (!results) {
    return (
      <div className="no-results">
        <h2>No Results Yet</h2>
        <p style={{ marginTop: '1rem' }}>
          Run a <Link to="/simulate">simulation</Link> first to see your retirement projections.
        </p>
      </div>
    );
  }

  const riskClass = `risk-${results.riskAssessment.toLowerCase()}`;

  return (
    <div>
      <h1 className="page-title">Your Retirement Plan Results</h1>

      <div className="card">
        <h3 style={{ color: '#059669', marginBottom: '1rem', letterSpacing: '-0.01em' }}>Overview</h3>
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{formatCurrency(results.totalSavingsAtRetirement)}</div>
            <div className="stat-label">Projected Savings at Retirement</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{formatCurrency(results.totalNeededForRetirement)}</div>
            <div className="stat-label">Total Needed for Retirement</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{formatCurrency(results.monthlyIncomeFromSavings)}</div>
            <div className="stat-label">Monthly Income from Savings</div>
          </div>
          <div className="stat-card">
            <div className={results.onTrack ? 'stat-value on-track' : 'stat-value off-track'}
                 style={{ WebkitTextFillColor: 'unset', background: 'none' }}>
              {results.onTrack ? 'ON TRACK' : 'OFF TRACK'}
            </div>
            <div className="stat-label">Status</div>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 style={{ color: '#059669', marginBottom: '1rem', letterSpacing: '-0.01em' }}>Key Metrics</h3>
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value" style={{ fontSize: '1.4rem' }}>{results.yearsToRetirement} yrs</div>
            <div className="stat-label">Years to Retirement</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ fontSize: '1.4rem' }}>{results.yearsInRetirement} yrs</div>
            <div className="stat-label">Years in Retirement</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ fontSize: '1.4rem' }}>{formatCurrency(results.totalContributions)}</div>
            <div className="stat-label">Total Contributions</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ fontSize: '1.4rem' }}>{formatCurrency(results.totalInterestEarned)}</div>
            <div className="stat-label">Interest Earned</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ fontSize: '1.4rem' }}>{formatCurrency(results.inflationAdjustedMonthlyIncome)}</div>
            <div className="stat-label">Inflation-Adjusted Monthly Need</div>
          </div>
          <div className="stat-card">
            <div className={`stat-value ${riskClass}`}
                 style={{ fontSize: '1.4rem', WebkitTextFillColor: 'unset', background: 'none' }}>
              {results.riskAssessment}
            </div>
            <div className="stat-label">Risk Level</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ fontSize: '1.4rem' }}>{formatCurrency(results.savingsGap)}</div>
            <div className="stat-label">Savings Gap</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ fontSize: '1.4rem' }}>{formatCurrency(results.requiredMonthlySavings)}</div>
            <div className="stat-label">Required Monthly Savings</div>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 style={{ color: '#059669', marginBottom: '1rem', letterSpacing: '-0.01em' }}>Savings Growth Projection</h3>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={results.yearlyProjections}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e7eeeb" />
              <XAxis dataKey="age" stroke="#94a3b8" label={{ value: 'Age', position: 'insideBottom', offset: -5 }} />
              <YAxis stroke="#94a3b8" tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
              <Tooltip
                formatter={(value) => [formatCurrency(value), '']}
                contentStyle={{ background: 'white', border: '1px solid #e7eeeb', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.06)' }}
              />
              <Area type="monotone" dataKey="endBalance" stroke="#059669" fill="url(#colorBalance)" strokeWidth={2} name="Balance" />
              <defs>
                <linearGradient id="colorBalance" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h3 style={{ color: '#059669', marginBottom: '1rem', letterSpacing: '-0.01em' }}>Recommendations</h3>
        <ul className="recommendations-list">
          {results.recommendations.map((rec, i) => (
            <li key={i}>{rec}</li>
          ))}
        </ul>
      </div>

      <div className="card">
        <h3 style={{ color: '#059669', marginBottom: '1rem', letterSpacing: '-0.01em' }}>Year-by-Year Projection</h3>
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
              {results.yearlyProjections.map((row, i) => (
                <tr key={i}>
                  <td>{row.age}</td>
                  <td>{formatCurrency(row.startBalance)}</td>
                  <td>{formatCurrency(row.contributions)}</td>
                  <td>{formatCurrency(row.interestEarned)}</td>
                  <td>{formatCurrency(row.endBalance)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default ResultsPage;
