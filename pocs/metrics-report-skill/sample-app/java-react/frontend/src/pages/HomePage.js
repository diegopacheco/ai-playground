import React from 'react';
import { Link } from 'react-router-dom';

function HomePage() {
  return (
    <div>
      <div className="hero-section">
        <h1 className="hero-title">Plan Your Retirement<br /><span>With Confidence</span></h1>
        <p className="hero-subtitle">
          Smart calculations, personalized recommendations, and clear projections
          to help you build the retirement you deserve.
        </p>
        <Link to="/simulate">
          <button className="btn-primary" style={{ width: 'auto', padding: '1rem 3rem', borderRadius: '14px' }}>
            Start Simulation
          </button>
        </Link>
      </div>
      <div className="features-grid">
        <div className="feature-card">
          <div className="feature-icon">$</div>
          <div className="feature-title">Compound Growth</div>
          <p style={{ color: '#64748b', lineHeight: '1.6', fontSize: '0.92rem' }}>
            See how your savings grow over time with compound interest calculations
            adjusted for your expected returns.
          </p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">%</div>
          <div className="feature-title">Inflation Adjusted</div>
          <p style={{ color: '#64748b', lineHeight: '1.6', fontSize: '0.92rem' }}>
            All projections account for inflation so you know the real purchasing
            power of your retirement savings.
          </p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">#</div>
          <div className="feature-title">Risk Assessment</div>
          <p style={{ color: '#64748b', lineHeight: '1.6', fontSize: '0.92rem' }}>
            Get a clear risk rating for your retirement plan with actionable
            recommendations to improve your outlook.
          </p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">~</div>
          <div className="feature-title">Year-by-Year View</div>
          <p style={{ color: '#64748b', lineHeight: '1.6', fontSize: '0.92rem' }}>
            Detailed yearly projections and interactive charts show your path
            to retirement savings goals.
          </p>
        </div>
      </div>
    </div>
  );
}

export default HomePage;
