export type Card = {
  id: number;
  symbol: string;
  matched: boolean;
};

const SYMBOLS = ["🍎", "🍌", "🍒", "🍇", "🍋", "🍓", "🍉", "🥝"];

export function buildDeck(): Card[] {
  const deck: Card[] = SYMBOLS.flatMap((symbol, i) => [
    { id: i * 2, symbol, matched: false },
    { id: i * 2 + 1, symbol, matched: false },
  ]);
  return shuffle(deck);
}

export function shuffle<T>(items: T[]): T[] {
  const copy = items.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

export function allMatched(deck: Card[]): boolean {
  return deck.every((c) => c.matched);
}
