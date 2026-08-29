import { test } from 'node:test'
import assert from 'node:assert/strict'
import { cartTotal } from '../src/cart.mjs'
import { charge } from '../src/checkout.mjs'

const cheapCart = [{ sku: 'MS-9', price: 15, qty: 1 }]
const coupon = { type: 'fixed', value: 25 }

test('a fixed coupon larger than the cart never makes the total negative', () => {
  assert.equal(cartTotal(cheapCart, coupon), 0)
})

test('checkout never charges a negative amount', () => {
  const { amountToCharge } = charge(cheapCart, coupon)
  assert.ok(amountToCharge >= 0, `amountToCharge was ${amountToCharge}`)
})
