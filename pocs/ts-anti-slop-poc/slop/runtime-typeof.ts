export function label(value: string | number): string {
  if (typeof value === "string") {
    return value;
  }

  return value.toFixed(2);
}
