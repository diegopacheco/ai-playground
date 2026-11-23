export function total(prices: readonly number[]): number {
  let sum = 0;
  for (const price of prices) {
    sum += price;
  }
  return sum;
}
