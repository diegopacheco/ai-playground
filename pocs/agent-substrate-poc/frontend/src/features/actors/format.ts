export function age(iso: string, now: number): string {
  const created = Date.parse(iso);
  if (Number.isNaN(created) || created <= 0) return "-";
  const s = Math.max(0, Math.floor((now - created) / 1000));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h`;
}
