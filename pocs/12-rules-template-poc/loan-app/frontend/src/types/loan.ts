export type AutoLoanInput = {
  amount: number
  termMonths: number
  annualIncome: number
  vehicleValue: number
  creditScore: number
}

export type AutoLoanDecision = {
  approved: boolean
  monthlyPayment: number
  interestRate: number
  reason: string
}
