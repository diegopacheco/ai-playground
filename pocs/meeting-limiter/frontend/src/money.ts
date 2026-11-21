export type Cents = number;

const formatter = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

export function formatMoney(cents: Cents): string {
  return formatter.format(cents / 100);
}

export function formatMinutes(minutes: number): string {
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  if (h === 0) return `${m}m`;
  return m === 0 ? `${h}h` : `${h}h ${m}m`;
}
