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

## Wave 3 — 2026-03-01

### What was tried
- Virtual threads: replaced Executors.newFixedThreadPool with Thread.ofVirtual()
- Primitive arrays for aggregation: replaced HashMap<String,Double> with double[] indexed by ordinal matching
- MemorySegment (Panama API): replaced MappedByteBuffer with java.lang.foreign.MemorySegment for direct memory access
- JVM tuning flags: -XX:+UseZGC -XX:+AlwaysPreTouch -XX:-TieredCompilation

### Results
| Metric | Before | After | Delta |
|---|---|---|---|
| Avg Time | 196.7ms | 225.0ms | +14.4% |
| Throughput | 165.7 MB/s | 144.9 MB/s | -12.5% |
| Rows/sec | 5,083,884 | 4,444,444 | -12.6% |

### Verdict: WORSE

### Why
- -XX:-TieredCompilation forces direct C2 compilation, adding startup overhead for short-lived JVM processes
- ZGC adds overhead for workloads with minimal GC pressure (primitive arrays produce no garbage)
- MemorySegment per-byte access via ValueLayout.JAVA_BYTE may have more overhead than ByteBuffer.get()
- Virtual threads provided no benefit — CPU-bound work doesn't benefit from lightweight scheduling
- Primitive array aggregation was a genuine improvement but masked by JVM flag regression

### PARTIAL ROLLBACK applied
Kept: primitive arrays + ordinal matching (no HashMap boxing). Reverted: JVM flags, MemorySegment (back to MappedByteBuffer), virtual threads (back to platform threads).

After rollback:
| Metric | Value | vs Wave 2 |
|---|---|---|
| Avg Time | 189.7ms | -3.6% |
| Throughput | 171.9 MB/s | +3.7% |
| Rows/sec | 5,271,481 | +3.7% |

Verdict after rollback: **BETTER** (marginal improvement from primitive array optimization)

## Wave 4 — 2026-03-01

### What was tried
- Bulk byte copy: copy 64KB chunks from MappedByteBuffer into local byte[] arrays, scan from local memory
- Branchless field scanning: streamlined comma-finding loop with early exit after 5 commas found
- Fixed-point arithmetic: replaced all double arithmetic with long-based fixed-point (4 decimal places)
- AOT compilation: SKIPPED (GraalVM native-image not available on this system)

### Results
| Metric | Before | After | Delta |
|---|---|---|---|
| Avg Time | 189.7ms | 151.3ms | -20.2% |
| Throughput | 171.9 MB/s | 215.5 MB/s | +25.4% |
| Rows/sec | 5,271,481 | 6,609,385 | +25.4% |

### Delta vs Baseline
| Metric | Baseline | Wave 4 | Delta |
|---|---|---|---|
| Avg Time | 411.7ms | 151.3ms | -63.3% |
| Throughput | 79.2 MB/s | 215.5 MB/s | +172.1% |
| Rows/sec | 2,428,953 | 6,609,385 | +172.0% |

### Verdict: BETTER

### Why
- Bulk byte copy eliminated per-byte ByteBuffer.get() overhead — copying 64KB at once is much cheaper than individual calls
- Fixed-point arithmetic replaced all floating-point multiply/add with integer operations — long math is faster and avoids FPU pipeline stalls
- Extracted processLine() into a separate method helped JIT focus optimization on the hot loop
- Minor revenue rounding difference (0.01 on 18B total) is acceptable fixed-point precision artifact

## Wave 5 — 2026-03-01

### What was tried
- Vectorized comma scan: SIMD ByteVector.SPECIES_256 to find commas 32 bytes at a time
- Larger bulk buffer: increased from 64KB to 256KB
- Direct FileInputStream: read entire file into byte[] instead of mmap
- Thread-local line buffer: process directly from shared byte[] without copying into line buffer

### Results
| Metric | Before | After | Delta |
|---|---|---|---|
| Avg Time | 151.3ms | 208.6ms | +37.9% |
| Throughput | 215.5 MB/s | 156.3 MB/s | -27.5% |
| Rows/sec | 6,609,385 | 4,793,864 | -27.5% |

### Verdict: WORSE

### Why
- FileInputStream.read() for entire 32MB file has higher upfront cost than mmap which lazily pages
- Vector API SIMD comma scan has JIT warmup overhead and doesn't benefit for short fields (5-10 bytes) — the scalar loop finishes before SIMD kicks in
- Allocating a 32MB byte[] adds GC pressure compared to OS-managed mmap pages
- The Vector API incubator module adds class loading overhead at startup
