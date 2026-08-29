import { round2 } from './money.mjs'

export function lineTotal(item) {
  return item.price * item.qty
}

export function subtotal(items) {
  return round2(items.reduce((sum, item) => sum + lineTotal(item), 0))
}

export function discountFor(sub, coupon) {
  if (!coupon) return 0
  if (coupon.type === 'percent') return round2(sub * (coupon.value / 100))
  return round2(coupon.value)
}

export function cartTotal(items, coupon) {
  const sub = subtotal(items)
  return round2(sub - discountFor(sub, coupon))
}
