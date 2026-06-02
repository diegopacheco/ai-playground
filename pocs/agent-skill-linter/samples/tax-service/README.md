# tax-service

A Java 25 + Spring Boot 4 service that computes income tax from progressive
brackets, deductions, and child tax credits. Built as a target codebase for the
`agent-skill-linter` skill.

## Tax rules

- Progressive brackets per filing status (single, married filing jointly, head of household).
- Standard deduction per status; itemized deduction wins when larger.
- Child tax credit of 2000 per dependent with income-based phase-out.
- Effective and marginal rates reported per return.

## Run

```
./start.sh
./stop.sh
```

Service listens on http://localhost:8080.

- `GET /api/tax/health`
- `POST /api/tax/calculate` with body `{ "filingStatus": "SINGLE", "grossIncome": 50000, "dependents": 0, "itemizedDeductions": 0 }`

## Test

```
./test.sh
```
