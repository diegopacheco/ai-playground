import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ResultsPage from '../pages/ResultsPage'

const mockResults = {
  totalSavingsAtRetirement: 1500000,
  totalNeededForRetirement: 1200000,
  monthlyIncomeFromSavings: 5000,
  savingsGap: 0,
  onTrack: true,
  yearsToRetirement: 35,
  yearsInRetirement: 25,
  totalContributions: 470000,
  totalInterestEarned: 1030000,
  inflationAdjustedMonthlyIncome: 14000,
  requiredMonthlySavings: 800,
  riskAssessment: 'LOW',
  yearlyProjections: [
    { year: 1, age: 31, startBalance: 50000, contributions: 12000, interestEarned: 3640, endBalance: 65640 },
  ],
  recommendations: ['You are well on track.'],
}

describe('ResultsPage', () => {
  it('renders status', () => {
    render(<ResultsPage results={mockResults} inputData={{}} onBack={() => {}} />)
    expect(screen.getByText('ON TRACK')).toBeInTheDocument()
  })

  it('renders risk level', () => {
    render(<ResultsPage results={mockResults} inputData={{}} onBack={() => {}} />)
    expect(screen.getByText('LOW')).toBeInTheDocument()
  })
})
