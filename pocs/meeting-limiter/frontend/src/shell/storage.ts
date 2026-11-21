export function readJson<T>(key: string, parse: (raw: unknown) => T | undefined, fallback: T): T {
  try {
    const text = window.localStorage.getItem(key);
    if (text === null) return fallback;
    return parse(JSON.parse(text)) ?? fallback;
  } catch {
    return fallback;
  }
}

export function writeJson(key: string, value: unknown): void {
  window.localStorage.setItem(key, JSON.stringify(value));
}
