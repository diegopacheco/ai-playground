export function readPort(raw: string): number {
  try {
    return Number.parseInt(raw, 10);
  } catch {
    return 0;
  }
}
