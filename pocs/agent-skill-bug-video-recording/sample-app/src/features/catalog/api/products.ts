export type Product = { id: string; name: string; price: number }

const products: Product[] = [
  { id: 'p1', name: 'Ceramic Pour-Over Coffee Dripper Cone Set', price: 24 },
  { id: 'p2', name: 'Stainless Steel Insulated Water Bottle 750ml', price: 32 },
  { id: 'p3', name: 'Hand-Poured Soy Wax Candle Lavender Fields', price: 18 },
  { id: 'p4', name: 'Organic Cotton Waffle-Knit Throw Blanket Large', price: 58 }
]

export function fetchProducts(): Promise<Product[]> {
  return new Promise(resolve => setTimeout(() => resolve(products), 150))
}
