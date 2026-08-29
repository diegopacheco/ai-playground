import { cartTotal } from './cart.mjs'
import { round2 } from './money.mjs'

const TAX_RATE = 0.08

export function charge(items, coupon) {
  const total = cartTotal(items, coupon)
  const tax = round2(total * TAX_RATE)
  return { total, tax, amountToCharge: round2(total + tax) }
}
