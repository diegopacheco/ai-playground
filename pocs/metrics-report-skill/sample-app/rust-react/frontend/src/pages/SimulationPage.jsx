import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'

const defaults = {
  currentAge: 30,
  retirementAge: 65,
  currentSavings: 50000,
  monthlyContribution: 1000,
  expectedAnnualReturn: 7,
  desiredMonthlyIncome: 5000,
  lifeExpectancy: 90,
  inflationRate: 3,
}

export default function SimulationPage({ onResults }) {
  const [form, setForm] = useState(defaults)
  const [error, setError] = useState(null)

  const mutation = useMutation({
    mutationFn: async (data) => {
      const res = await fetch('http://localhost:8080/api/retirement/calculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
      const json = await res.json()
      if (!res.ok) throw new Error(json.error || 'Calculation failed')
      return json
    },
    onSuccess: (data) => {
      setError(null)
      onResults(data, form)
    },
    onError: (err) => setError(err.message),
  })

  function handleChange(e) {
    setForm({ ...form, [e.target.name]: parseFloat(e.target.value) || 0 })
  }

  function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    mutation.mutate(form)
  }

  return (
    <div className="card">
      <h2 style={{ marginBottom: 24, fontSize: '1.5rem' }}>Retirement Simulation</h2>
      {error && <div className="error-message">{error}</div>}
      <form onSubmit={handleSubmit}>
        <div className="form-section">
          <h3>Personal Information</h3>
          <div className="form-grid">
            <div className="form-group">
              <label>Current Age</label>
              <input type="number" name="currentAge" value={form.currentAge} onChange={handleChange} />
            </div>
            <div className="form-group">
              <label>Retirement Age</label>
              <input type="number" name="retirementAge" value={form.retirementAge} onChange={handleChange} />
            </div>
            <div className="form-group">
              <label>Life Expectancy</label>
              <input type="number" name="lifeExpectancy" value={form.lifeExpectancy} onChange={handleChange} />
            </div>
          </div>
        </div>
        <div className="form-section">
          <h3>Financial Details</h3>
          <div className="form-grid">
            <div className="form-group">
              <label>Current Savings ($)</label>
              <input type="number" name="currentSavings" value={form.currentSavings} onChange={handleChange} />
            </div>
            <div className="form-group">
              <label>Monthly Contribution ($)</label>
              <input type="number" name="monthlyContribution" value={form.monthlyContribution} onChange={handleChange} />
            </div>
            <div className="form-group">
              <label>Desired Monthly Income ($)</label>
              <input type="number" name="desiredMonthlyIncome" value={form.desiredMonthlyIncome} onChange={handleChange} />
            </div>
          </div>
        </div>
        <div className="form-section">
          <h3>Market Assumptions</h3>
          <div className="form-grid">
            <div className="form-group">
              <label>Expected Annual Return (%)</label>
              <input type="number" step="0.1" name="expectedAnnualReturn" value={form.expectedAnnualReturn} onChange={handleChange} />
            </div>
            <div className="form-group">
              <label>Inflation Rate (%)</label>
              <input type="number" step="0.1" name="inflationRate" value={form.inflationRate} onChange={handleChange} />
            </div>
          </div>
        </div>
        <button type="submit" className="btn-primary" disabled={mutation.isPending}>
          {mutation.isPending ? 'Calculating...' : 'Calculate'}
        </button>
      </form>
    </div>
  )
}
