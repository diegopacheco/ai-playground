export function publishedDay(published: string): string {
  return published.slice(0, 10);
}

export function excerpt(text: string, max: number): string {
  if (text.length <= max) {
    return text;
  }
  const cut = text.slice(0, max);
  const lastSpace = cut.lastIndexOf(" ");
  return `${cut.slice(0, lastSpace > 0 ? lastSpace : max)}…`;
}

export function matchPercent(score: number): string {
  return `${Math.round(score * 100)}% match`;
}
