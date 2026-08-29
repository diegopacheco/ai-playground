import { test } from 'node:test'
import assert from 'node:assert/strict'
import { subtotal, cartTotal } from '../src/cart.mjs'
import { charge } from '../src/checkout.mjs'

const items = [
  { sku: 'KB-1', price: 42.5, qty: 2 },
  { sku: 'MS-9', price: 15, qty: 1 },
]

test('subtotal adds every line', () => {
  assert.equal(subtotal(items), 100)
})

test('percent coupon takes its share', () => {
  assert.equal(cartTotal(items, { type: 'percent', value: 10 }), 90)
})

test('fixed coupon takes its amount', () => {
  assert.equal(cartTotal(items, { type: 'fixed', value: 20 }), 80)
})

test('checkout adds tax on the total', () => {
  assert.deepEqual(charge(items, null), { total: 100, tax: 8, amountToCharge: 108 })
})
