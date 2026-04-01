import { Link } from '@tanstack/react-router'

const features = [
  { title: 'Compound Growth', desc: 'Monthly compounding with configurable annual return rates for accurate projections.' },
  { title: 'Inflation Adjustment', desc: 'All projections account for inflation to show real purchasing power.' },
  { title: 'Risk Assessment', desc: 'Four-tier risk rating based on your savings-to-need ratio.' },
  { title: 'Year-by-Year View', desc: 'Detailed projections with interactive charts showing balance growth.' },
]

export default function HomePage() {
  return (
    <div>
      <div className="hero">
        <h1>Plan Your <span>Retirement</span></h1>
        <p>
          Project your financial future with detailed calculations, risk assessment, and personalized recommendations.
        </p>
        <Link to="/simulate" className="btn-primary">Start Simulation</Link>
      </div>
      <div className="features-grid">
        {features.map((f) => (
          <div key={f.title} className="feature-card">
            <div className="feature-icon">$</div>
            <h3>{f.title}</h3>
            <p>{f.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
