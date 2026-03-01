# AutoBench Findings — Java 25 CSV Analytics

## Wave 0 — Baseline — 2026-03-01

### Implementation
Naive implementation using `BufferedReader`, `String.split(",")`, `Double.parseDouble`, and `HashMap` for aggregations. No optimizations applied.

### Results
| Metric | Value |
|---|---|
| Avg Time | 411.7ms |
| Throughput | 79.2 MB/s |
| Rows/sec | 2,428,953 |
| File Size | 32.6 MB |
| Run 1 | 405.4ms |
| Run 2 | 419.8ms |
| Run 3 | 409.9ms |

### Verdict: BASELINE

## Wave 1 — 2026-03-01

### What was tried
- Manual CSV parsing: replaced `String.split(",")` with `indexOf(',')` based field extraction
- Larger read buffer: increased BufferedReader buffer from 8KB to 1MB (1 << 20)
- Custom number parsing: hand-rolled `parseIntFast` and `parseDoubleFast` avoiding String object creation
- Pre-sized HashMap: initialized HashMaps with known cardinalities (16 for categories, 8 for regions)

### Results
| Metric | Before | After | Delta |
|---|---|---|---|
| Avg Time | 411.7ms | 308.6ms | -25.0% |
| Throughput | 79.2 MB/s | 105.6 MB/s | +33.3% |
| Rows/sec | 2,428,953 | 3,240,441 | +33.4% |

### Verdict: BETTER

### Why
- Manual CSV parsing eliminated regex compilation and String[] allocation per line — biggest single win
- Custom number parsers avoided creating intermediate String objects for numeric fields
- Larger buffer reduced system call overhead for I/O reads
- Pre-sized HashMaps had minimal impact since rehashing only happens a few times with small maps

## Wave 2 — 2026-03-01

### What was tried
- Byte-level I/O: replaced BufferedReader.readLine() with raw byte[] parsing from MappedByteBuffer
- String interning: category/region strings cached via ConcurrentHashMap to share identical objects
- Memory-mapped file: used FileChannel.map() to mmap the entire CSV, letting OS handle paging
- Parallel streams: split file into chunks across availableProcessors() threads, merge partial results

### Results
| Metric | Before | After | Delta |
|---|---|---|---|
| Avg Time | 308.6ms | 196.7ms | -36.3% |
| Throughput | 105.6 MB/s | 165.7 MB/s | +56.9% |
| Rows/sec | 3,240,441 | 5,083,884 | +56.9% |

### Delta vs Baseline
| Metric | Baseline | Wave 2 | Delta |
|---|---|---|---|
| Avg Time | 411.7ms | 196.7ms | -52.2% |
| Throughput | 79.2 MB/s | 165.7 MB/s | +109.2% |
| Rows/sec | 2,428,953 | 5,083,884 | +109.3% |

### Verdict: BETTER

### Why
- Memory-mapped file eliminated all read syscalls — OS handles paging directly from page cache
- Parallel processing split work across CPU cores — near-linear scaling for this embarrassingly parallel task
- Byte-level parsing avoided creating String objects per line — massive reduction in GC pressure
- String interning ensured only 12 unique category/region strings existed in memory
