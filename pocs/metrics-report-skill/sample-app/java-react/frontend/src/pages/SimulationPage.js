import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const API_URL = 'http://localhost:8080/api/retirement';

function SimulationPage({ setResults, setInputData }) {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [form, setForm] = useState({
    currentAge: 30,
    retirementAge: 65,
    currentSavings: 50000,
    monthlyContribution: 1000,
    expectedAnnualReturn: 7,
    desiredMonthlyIncome: 5000,
    lifeExpectancy: 90,
    inflationRate: 3
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: parseFloat(e.target.value) || '' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/calculate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form)
      });
      const data = await response.json();
      if (!response.ok) {
        const msg = data.error || (data.errors ? Object.values(data.errors).join(', ') : 'Calculation failed');
        setError(msg);
        return;
      }
      setResults(data);
      setInputData(form);
      navigate('/results');
    } catch (err) {
      setError('Cannot connect to server. Make sure the backend is running on port 8080.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 className="page-title">Retirement Simulation</h1>
      <form onSubmit={handleSubmit}>
        <div className="card">
          <h3 style={{ color: '#059669', marginBottom: '1rem', letterSpacing: '-0.01em' }}>Personal Information</h3>
          <div className="form-grid">
            <div className="form-group">
              <label>Current Age</label>
              <input type="number" name="currentAge" value={form.currentAge} onChange={handleChange} min="18" max="80" />
            </div>
            <div className="form-group">
              <label>Retirement Age</label>
              <input type="number" name="retirementAge" value={form.retirementAge} onChange={handleChange} min="50" max="75" />
            </div>
            <div className="form-group">
              <label>Life Expectancy</label>
              <input type="number" name="lifeExpectancy" value={form.lifeExpectancy} onChange={handleChange} min="70" max="110" />
            </div>
          </div>
        </div>
        <div className="card">
          <h3 style={{ color: '#059669', marginBottom: '1rem', letterSpacing: '-0.01em' }}>Financial Details</h3>
          <div className="form-grid">
            <div className="form-group">
              <label>Current Savings ($)</label>
              <input type="number" name="currentSavings" value={form.currentSavings} onChange={handleChange} min="0" />
            </div>
            <div className="form-group">
              <label>Monthly Contribution ($)</label>
              <input type="number" name="monthlyContribution" value={form.monthlyContribution} onChange={handleChange} min="0" max="100000" />
            </div>
            <div className="form-group">
              <label>Desired Monthly Income in Retirement ($)</label>
              <input type="number" name="desiredMonthlyIncome" value={form.desiredMonthlyIncome} onChange={handleChange} min="500" />
            </div>
          </div>
        </div>
        <div className="card">
          <h3 style={{ color: '#059669', marginBottom: '1rem', letterSpacing: '-0.01em' }}>Market Assumptions</h3>
          <div className="form-grid">
            <div className="form-group">
              <label>Expected Annual Return (%)</label>
              <input type="number" name="expectedAnnualReturn" value={form.expectedAnnualReturn} onChange={handleChange} min="0" max="30" step="0.1" />
            </div>
            <div className="form-group">
              <label>Inflation Rate (%)</label>
              <input type="number" name="inflationRate" value={form.inflationRate} onChange={handleChange} min="0" max="15" step="0.1" />
            </div>
          </div>
        </div>
        {error && <div className="error-text" style={{ marginBottom: '1rem', fontSize: '1rem' }}>{error}</div>}
        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Calculating...' : 'Calculate Retirement Plan'}
        </button>
      </form>
    </div>
  );
}

export default SimulationPage;
