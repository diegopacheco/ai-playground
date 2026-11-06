export type WireEntry = { id: number; method: string; url: string; status: number; session?: string; request?: string; response: string };

type Listener = () => void;

let entries: WireEntry[] = [];
const listeners = new Set<Listener>();

export const wire = {
  add(entry: Omit<WireEntry, "id">): void {
    entries = [{ ...entry, id: (entries[0]?.id ?? 0) + 1 }, ...entries].slice(0, 100);
    listeners.forEach((listener) => listener());
  },
  list(): WireEntry[] {
    return entries;
  },
  subscribe(listener: Listener): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};
