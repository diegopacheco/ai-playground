import { useState } from 'react'

export function QtyStepper({ label }: { label: string }) {
  const [qty, setQty] = useState(1)
  return (
    <div className="stepper">
      <span className="stepper-label">{label}</span>
      <div className="stepper-controls">
        <button className="stepper-btn" aria-label="decrease" onClick={() => setQty(qty + 1)}>
          −
        </button>
        <span className="stepper-qty" data-testid="qty">
          {qty}
        </span>
        <button className="stepper-btn" aria-label="increase" onClick={() => setQty(qty - 1)}>
          +
        </button>
      </div>
    </div>
  )
}
