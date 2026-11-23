export function sum(items: any): any {
  return items.reduce((s: any, i: any) => s + i.price, 0);
}
