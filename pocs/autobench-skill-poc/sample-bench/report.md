# AutoBench Final Report — Java 25 CSV Analytics

**Date:** 2026-03-01
**Language:** Java 25 (Corretto 25.0.0)
**Benchmark:** CSV Analytics — 1M rows, 32.6MB file
**Waves:** 5 (2 rolled back)

## Summary Table

| Wave | Optimizations | Avg Time | Throughput | Rows/sec | Delta vs Baseline | Verdict |
|---|---|---|---|---|---|---|
| 0 (Baseline) | None — BufferedReader + split + HashMap | 411.7ms | 79.2 MB/s | 2,428,953 | — | BASELINE |
| 1 | Manual parsing, 1MB buffer, custom number parse, pre-sized maps | 308.6ms | 105.6 MB/s | 3,240,441 | -25.0% | BETTER |
| 2 | Mmap, multithreaded, byte-level I/O, string interning | 196.7ms | 165.7 MB/s | 5,083,884 | -52.2% | BETTER |
| 3 | Virtual threads, MemorySegment, primitive arrays, ZGC flags | 225.0ms | 144.9 MB/s | 4,444,444 | -45.3% | WORSE (partial rollback) |
| 3r | Kept primitive arrays, reverted JVM flags/MemorySegment | 189.7ms | 171.9 MB/s | 5,271,481 | -53.9% | BETTER |
| 4 | Bulk byte copy (64KB), fixed-point arithmetic, processLine extraction | 151.3ms | 215.5 MB/s | 6,609,385 | -63.3% | BETTER |
| 5 | SIMD Vector API, FileInputStream, 256KB buffer | 208.6ms | 156.3 MB/s | 4,793,864 | -49.3% | WORSE (rolled back) |

## Performance Trend

```
Avg Time (ms)
  420 |X
      |
  380 |
      |
  340 |
      |
  300 |  X
      |
  260 |
      |
  220 |        x(W3 pre-rollback)
      |
  200 |     X        x(W5 pre-rollback)
      |
  160 |       X(W3r)   X(W4 BEST)
      |
  120 |
      |
   80 |
      +-----+-----+-----+-----+-----+---
       W0    W1    W2    W3    W4    W5

  X = kept version    x = rolled back version
```

## Top Findings — Impact Ranking

| Rank | Optimization | Impact | Wave |
|---|---|---|---|
| 1 | Memory-mapped file (mmap) | Eliminated all read syscalls | W2 |
| 2 | Multi-threaded chunk processing | Near-linear scaling across cores | W2 |
| 3 | Bulk byte copy (64KB chunks) | Eliminated per-byte ByteBuffer.get() overhead | W4 |
| 4 | Fixed-point arithmetic (long) | Removed all floating-point from hot loop | W4 |
| 5 | Manual CSV parsing (indexOf) | Eliminated regex + array alloc per line | W1 |
| 6 | Custom number parsers | Avoided String creation for numeric fields | W1 |
| 7 | Primitive array aggregation | Eliminated HashMap boxing overhead | W3r |
| 8 | Ordinal-based category matching | Single byte switch vs string comparison | W3r |

## What Did NOT Help

| Optimization | Why It Hurt | Wave |
|---|---|---|
| -XX:-TieredCompilation | Forces C2 upfront — terrible for short-lived JVM | W3 |
| -XX:+UseZGC | Overhead exceeds benefit when GC pressure is near zero | W3 |
| MemorySegment (Panama) | Per-byte access slower than ByteBuffer for this pattern | W3 |
| Virtual threads | CPU-bound work gets no benefit from lightweight scheduling | W3 |
| SIMD Vector API (comma scan) | Fields too short (5-10 bytes) for 32-byte SIMD to win | W5 |
| Full-file byte[] read | 32MB allocation + copy slower than lazy mmap paging | W5 |

## Final Recommendation

**Best configuration: Wave 4** — 151.3ms average, 215.5 MB/s throughput, 6.6M rows/sec

Key techniques in the winning version:
- **MappedByteBuffer** for zero-copy file access via OS page cache
- **Platform threads** (one per CPU core) for parallel chunk processing
- **64KB bulk byte copy** from mmap into local arrays for cache-friendly scanning
- **Fixed-point long arithmetic** (4 decimal places) instead of double
- **Primitive array aggregation** with ordinal-based category/region matching
- **Custom byte-level number parsers** avoiding all String object creation

**Total improvement: 2.72x faster** (411.7ms → 151.3ms, or 63.3% reduction)
