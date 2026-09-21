import { describe, expect, test } from "bun:test";
import { formatLatency, formatSimilarity, hitRate } from "./format";

describe("format", () => {
  test("similarity reads as a percentage so it compares against the threshold at a glance", () => {
    expect(formatSimilarity(0.9641)).toBe("96.4%");
    expect(formatSimilarity(null)).toBe("no neighbor");
  });

  test("slow Claude calls read in seconds and cache hits in milliseconds", () => {
    expect(formatLatency(42)).toBe("42 ms");
    expect(formatLatency(8250)).toBe("8.3 s");
  });

  test("hit rate is zero before any question instead of NaN", () => {
    expect(hitRate({ hits: 0, misses: 0 })).toBe("0%");
    expect(hitRate({ hits: 3, misses: 1 })).toBe("75%");
  });
});
