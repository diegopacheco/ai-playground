import type { AutoLoanDecision } from '../types/loan'

type Props = { decision: AutoLoanDecision }

export function LoanResult({ decision }: Props) {
  return (
    <div className={`result ${decision.approved ? 'approved' : 'denied'}`}>
      <div className="status">{decision.approved ? 'Approved' : 'Denied'}</div>
      <div className="reason">{decision.reason}</div>
      <dl className="terms">
        <dt>Monthly payment</dt>
        <dd>${decision.monthlyPayment.toFixed(2)}</dd>
        <dt>Interest rate</dt>
        <dd>{decision.interestRate.toFixed(2)}%</dd>
      </dl>
    </div>
  )
}
