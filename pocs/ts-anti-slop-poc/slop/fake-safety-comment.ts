type Order = { readonly total: number };

export function toOrder(raw: string): Order {
  const parsed = JSON.parse(raw);

  // SAFETY: trust me
  return parsed as Order;
}
