import type { Stats } from "../types";

export function formatSimilarity(value: number | null): string {
  return value === null ? "no neighbor" : `${(value * 100).toFixed(1)}%`;
}

export function formatLatency(ms: number): string {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

export function hitRate(stats: Pick<Stats, "hits" | "misses">): string {
  const total = stats.hits + stats.misses;
  return total === 0 ? "0%" : `${Math.round((stats.hits / total) * 100)}%`;
}

export function formatTime(millis: number): string {
  return new Date(millis).toLocaleString();
}
