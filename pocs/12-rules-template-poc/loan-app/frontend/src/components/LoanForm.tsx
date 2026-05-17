import { useState, type FormEvent } from 'react'
import type { AutoLoanInput } from '../types/loan'

type Props = {
  loading: boolean
  onSubmit: (input: AutoLoanInput) => void
}

const INITIAL: AutoLoanInput = {
  amount: 25000,
  termMonths: 60,
  annualIncome: 80000,
  vehicleValue: 30000,
  creditScore: 720
}

export function LoanForm({ loading, onSubmit }: Props) {
  const [form, setForm] = useState<AutoLoanInput>(INITIAL)

  function update<K extends keyof AutoLoanInput>(key: K, value: AutoLoanInput[K]) {
    setForm(prev => ({ ...prev, [key]: value }))
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    onSubmit(form)
  }

  return (
    <form className="form" onSubmit={handleSubmit}>
      <Field label="Loan amount ($)" value={form.amount} step={500}
             onChange={v => update('amount', v)} />
      <Field label="Term (months)" value={form.termMonths} step={1}
             onChange={v => update('termMonths', v)} />
      <Field label="Annual income ($)" value={form.annualIncome} step={1000}
             onChange={v => update('annualIncome', v)} />
      <Field label="Vehicle value ($)" value={form.vehicleValue} step={500}
             onChange={v => update('vehicleValue', v)} />
      <Field label="Credit score" value={form.creditScore} step={1}
             onChange={v => update('creditScore', v)} />
      <button type="submit" disabled={loading}>
        {loading ? 'Submitting…' : 'Request loan'}
      </button>
    </form>
  )
}

type FieldProps = {
  label: string
  value: number
  step: number
  onChange: (v: number) => void
}

function Field({ label, value, step, onChange }: FieldProps) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
      />
    </label>
  )
}
