import { PLUSH_LOOKS } from "./plush";

export type Collection = Readonly<Record<string, number>>;

export type KeyValueStore = Pick<Storage, "getItem" | "setItem">;

export type CollectionEntry = {
  readonly look: number;
  readonly name: string;
  readonly emoji: string;
  readonly count: number;
};

export const COLLECTION_KEY = "lucky-claw.collection";

export const EMPTY_COLLECTION: Collection = {};

export function addWin(collection: Collection, look: number): Collection {
  const name = PLUSH_LOOKS[look]?.name;
  if (!name) return collection;
  return { ...collection, [name]: (collection[name] ?? 0) + 1 };
}

export function totalWon(collection: Collection): number {
  return Object.values(collection).reduce((sum, count) => sum + count, 0);
}

export function collectionEntries(collection: Collection): readonly CollectionEntry[] {
  return PLUSH_LOOKS.flatMap((look, index) => {
    const count = collection[look.name] ?? 0;
    return count > 0 ? [{ look: index, name: look.name, emoji: look.emoji, count }] : [];
  });
}

function parse(raw: string): Collection {
  const data: unknown = JSON.parse(raw);
  if (typeof data !== "object" || data === null || Array.isArray(data)) return EMPTY_COLLECTION;
  const counts: Record<string, number> = {};
  for (const look of PLUSH_LOOKS) {
    const count = (data as Record<string, unknown>)[look.name];
    if (typeof count === "number" && Number.isInteger(count) && count > 0) counts[look.name] = count;
  }
  return counts;
}

export function loadCollection(store: KeyValueStore | undefined): Collection {
  try {
    const raw = store?.getItem(COLLECTION_KEY);
    return raw ? parse(raw) : EMPTY_COLLECTION;
  } catch {
    return EMPTY_COLLECTION;
  }
}

export function saveCollection(store: KeyValueStore | undefined, collection: Collection): void {
  try {
    store?.setItem(COLLECTION_KEY, JSON.stringify(collection));
  } catch {
    return;
  }
}

export function browserStore(): KeyValueStore | undefined {
  try {
    return globalThis.localStorage;
  } catch {
    return undefined;
  }
}
