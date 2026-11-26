export function percent(p: number): string {
  return `${Math.round(p * 100)}%`;
}

export function decimal(p: number): string {
  return p.toFixed(2);
}
