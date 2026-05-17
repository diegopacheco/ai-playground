import { useMutation } from '@tanstack/react-query'
import { LoanForm } from './components/LoanForm'
import { LoanResult } from './components/LoanResult'
import { requestAutoLoan } from './api/mutations'

export default function App() {
  const mutation = useMutation({ mutationFn: requestAutoLoan })

  return (
    <div className="app">
      <header>
        <h1>Auto Loan</h1>
        <p className="sub">Request financing for your next vehicle.</p>
      </header>

      <LoanForm loading={mutation.isPending} onSubmit={mutation.mutate} />

      {mutation.isError && (
        <div className="error">Error: {(mutation.error as Error).message}</div>
      )}
      {mutation.data && <LoanResult decision={mutation.data} />}
    </div>
  )
}
