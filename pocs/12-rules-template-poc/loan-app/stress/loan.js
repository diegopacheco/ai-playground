import http from 'k6/http'
import { check } from 'k6'

export const options = {
  stages: [
    { duration: '10s', target: 10 },
    { duration: '20s', target: 30 },
    { duration: '10s', target: 0 }
  ],
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<500']
  }
}

const MUTATION = 'mutation Req($i:AutoLoanInput!){requestAutoLoan(input:$i){approved monthlyPayment interestRate reason}}'

export default function () {
  const body = JSON.stringify({
    query: MUTATION,
    variables: {
      i: {
        amount: 25000,
        termMonths: 60,
        annualIncome: 80000,
        vehicleValue: 30000,
        creditScore: 720
      }
    }
  })

  const res = http.post('http://localhost:8080/graphql', body, {
    headers: { 'Content-Type': 'application/json' }
  })

  check(res, {
    'status 200': r => r.status === 200,
    'approved': r => r.json('data.requestAutoLoan.approved') === true,
    'no graphql errors': r => !r.json('errors')
  })
}
