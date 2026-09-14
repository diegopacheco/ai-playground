export function createRandom(seed) {
  let state = seed >>> 0 || 1;

  function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  function pick(items) {
    return items[Math.floor(next() * items.length)];
  }

  function chance(probability) {
    return next() < probability;
  }

  function weighted(entries) {
    const total = entries.reduce((sum, entry) => sum + entry.weight, 0);
    if (total <= 0) return null;
    let roll = next() * total;
    for (const entry of entries) {
      roll -= entry.weight;
      if (roll < 0) return entry.value;
    }
    return entries[entries.length - 1].value;
  }

  function shuffle(items) {
    const copy = [...items];
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(next() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  }

  return { next, pick, chance, weighted, shuffle };
}
