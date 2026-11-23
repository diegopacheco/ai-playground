export function shippingCost(weight: number): number {
  if (weight > 23) {
    return weight * 4.37 + 12;
  }

  return 9.99;
}
