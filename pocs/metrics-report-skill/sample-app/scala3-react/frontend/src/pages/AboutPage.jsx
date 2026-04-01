function AboutPage() {
  return (
    <div>
      <h1 className="page-title">About RetireSmart</h1>
      <div className="card about-content">
        <h3>What is RetireSmart?</h3>
        <p>
          RetireSmart is a retirement planning simulation tool that helps you project
          your financial future. By entering your current financial situation and goals,
          you get detailed projections showing whether you are on track for a comfortable retirement.
        </p>

        <h3>How Calculations Work</h3>
        <p>
          The simulation uses compound interest formulas to project your savings growth
          over time. Monthly contributions are added and interest is compounded monthly
          based on your expected annual return rate.
        </p>
        <p>
          Inflation is factored in to show the real purchasing power of your future
          savings. The desired monthly income is adjusted for inflation to determine
          how much you actually need saved at retirement.
        </p>

        <h3>Risk Assessment</h3>
        <p>
          Your plan receives a risk rating based on the ratio of projected savings
          to total needed funds:
        </p>
        <ul style={{ paddingLeft: '1.5rem', marginBottom: '1rem' }}>
          <li><span className="risk-low">LOW</span> - Projected savings exceed 150% of needed amount</li>
          <li><span className="risk-moderate">MODERATE</span> - Projected savings meet or exceed 100% of needed amount</li>
          <li><span className="risk-high">HIGH</span> - Projected savings cover 70-100% of needed amount</li>
          <li><span className="risk-critical">CRITICAL</span> - Projected savings cover less than 70% of needed amount</li>
        </ul>

        <h3>Validation Rules</h3>
        <p>The following business rules are enforced:</p>
        <ul style={{ paddingLeft: '1.5rem', marginBottom: '1rem' }}>
          <li>Age must be between 18 and 80</li>
          <li>Retirement age must be between 50 and 75</li>
          <li>At least 5 years between current age and retirement age</li>
          <li>Life expectancy must exceed retirement age</li>
          <li>Expected return must exceed inflation rate</li>
          <li>Monthly contribution cannot exceed $100,000</li>
          <li>Desired monthly income must be at least $500</li>
        </ul>

        <h3>Technology Stack</h3>
        <p>
          Built with Scala 3 and Spring Boot 4.0.4 on the backend for robust financial
          calculations and validation. The frontend uses React 19 with TanStack Router,
          Vite, Bun, and Recharts for interactive data visualization.
        </p>
      </div>
    </div>
  )
}

export default AboutPage
