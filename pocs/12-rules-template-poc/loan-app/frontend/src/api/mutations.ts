import { graphql } from './client'
import type { AutoLoanDecision, AutoLoanInput } from '../types/loan'

const REQUEST_AUTO_LOAN = `
  mutation RequestAutoLoan($input: AutoLoanInput!) {
    requestAutoLoan(input: $input) {
      approved
      monthlyPayment
      interestRate
      reason
    }
  }
`

export async function requestAutoLoan(input: AutoLoanInput): Promise<AutoLoanDecision> {
  const data = await graphql<{ requestAutoLoan: AutoLoanDecision }>(
    REQUEST_AUTO_LOAN,
    { input }
  )
  return data.requestAutoLoan
}
