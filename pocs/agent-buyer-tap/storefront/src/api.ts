export type Product = {
  sku: string;
  name: string;
  priceCents: number;
  currency: string;
  stock: number;
  tagline: string;
  accent: string;
};

export type Order = {
  id: string;
  buyer: string;
  items: { sku: string; qty: number }[];
  amountCents: number;
  currency: string;
  status: string;
  createdAt: string;
  agentKeyid: string;
  walletKid: string;
};

export async function getCatalog(): Promise<Product[]> {
  const res = await fetch('/api/catalog');
  const body = await res.json();
  return body.products;
}

export async function getOrder(id: string): Promise<Order> {
  const res = await fetch(`/api/orders/${id}`);
  if (!res.ok) throw new Error('order not found');
  return res.json();
}

export function money(cents: number, currency: string): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(cents / 100);
}
