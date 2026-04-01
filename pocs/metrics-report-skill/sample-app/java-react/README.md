# RetireSmart - Retirement Plan Simulation

A full-stack retirement planning simulation application that helps users project their financial future with detailed calculations, risk assessment, and personalized recommendations.

## Tech Stack

| Layer    | Technology              |
|----------|------------------------|
| Backend  | Java 25, Spring Boot 4.0.4 |
| Frontend | Node.js 24, React 19   |
| Charts   | Recharts               |
| Routing  | React Router v7        |

## Features

- **Compound Growth Simulation** - Monthly compounding with configurable annual return rates
- **Inflation Adjustment** - All projections factor in inflation to show real purchasing power
- **Risk Assessment** - Four-tier risk rating (LOW, MODERATE, HIGH, CRITICAL) based on savings-to-need ratio
- **Year-by-Year Projections** - Detailed table and interactive area chart showing balance growth
- **Smart Recommendations** - Personalized suggestions based on your financial profile
- **Input Validation** - Both client-side and server-side validation with business rule enforcement

## Pages

1. **Home** - Landing page with feature overview
2. **Simulate** - Input form with personal, financial, and market assumption fields
3. **Results** - Dashboard with stats, charts, projections table, and recommendations
4. **About** - Methodology, risk levels, and validation rules explained

## Business Rules

- Current age: 18-80
- Retirement age: 50-75
- Minimum 5 years gap between current age and retirement age
- Life expectancy must exceed retirement age
- Expected annual return must exceed inflation rate
- Monthly contribution cap: $100,000
- Minimum desired monthly income: $500

## API Endpoints

| Method | Endpoint                    | Description              |
|--------|-----------------------------|--------------------------|
| POST   | `/api/retirement/calculate` | Run retirement simulation |
| GET    | `/api/retirement/health`    | Health check             |

### Request Body (POST /api/retirement/calculate)

```json
{
  "currentAge": 30,
  "retirementAge": 65,
  "currentSavings": 50000,
  "monthlyContribution": 1000,
  "expectedAnnualReturn": 7,
  "desiredMonthlyIncome": 5000,
  "lifeExpectancy": 90,
  "inflationRate": 3
}
```

## Prerequisites

- Java 25
- Node.js 24
- Maven (wrapper included)

## Running

Start both backend and frontend:
```bash
./run.sh
```

Stop both:
```bash
./stop.sh
```

- Backend runs on http://localhost:8080
- Frontend runs on http://localhost:3000

## Running Tests

Backend tests:
```bash
cd backend && ./mvnw test
```

Frontend tests:
```bash
cd frontend && npm test
```
