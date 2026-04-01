export default function AboutPage() {
  return (
    <div className="card">
      <div className="about-section">
        <h2>What is RetireSmart?</h2>
        <p>
          RetireSmart is a retirement planning simulation tool that helps you project your financial
          future with detailed calculations, risk assessment, and personalized recommendations.
        </p>
      </div>

      <div className="about-section">
        <h2>How Calculations Work</h2>
        <ul>
          <li>Monthly compounding is used to simulate investment growth over your working years.</li>
          <li>Inflation is factored into your desired retirement income to reflect future purchasing power.</li>
          <li>A sustainable withdrawal rate (60% of expected return) determines your projected monthly income in retirement.</li>
          <li>The savings gap shows any shortfall between projected savings and total retirement needs.</li>
        </ul>
      </div>

      <div className="about-section">
        <h2>Risk Assessment</h2>
        <ul>
          <li><strong>LOW</strong> - Projected savings are at least 150% of what you need.</li>
          <li><strong>MODERATE</strong> - Projected savings meet or exceed your needs.</li>
          <li><strong>HIGH</strong> - Projected savings cover 70-100% of your needs.</li>
          <li><strong>CRITICAL</strong> - Projected savings cover less than 70% of your needs.</li>
        </ul>
      </div>

      <div className="about-section">
        <h2>Validation Rules</h2>
        <ul>
          <li>Current age: 18-80</li>
          <li>Retirement age: 50-75</li>
          <li>Minimum 5 years between current age and retirement age</li>
          <li>Life expectancy must exceed retirement age (range: 70-110)</li>
          <li>Expected annual return must exceed inflation rate</li>
          <li>Monthly contribution: $0-$100,000</li>
          <li>Minimum desired monthly income: $500</li>
        </ul>
      </div>

      <div className="about-section">
        <h2>Technology Stack</h2>
        <ul>
          <li><strong>Backend:</strong> Rust 1.93+ with Actix-web</li>
          <li><strong>Frontend:</strong> React 19, Vite, TanStack Router, TanStack Query</li>
          <li><strong>Charts:</strong> Recharts</li>
        </ul>
      </div>
    </div>
  )
}
