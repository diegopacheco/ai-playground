# Autobench in Action

## What is AutoBench?

AutoBench is a Claude Code skill that runs automated, iterative performance benchmarking on code. You pick a language, a benchmark type, and a number of optimization waves. AutoBench then generates a naive baseline implementation, measures it, and runs wave after wave of optimizations — each time proposing changes, letting you approve them, benchmarking the result, and honestly recording whether things got better or worse.

### How it works

1. **Setup** — You choose a language (Java, Go, Rust, Zig, Scala, TypeScript), a benchmark type (CSV Analytics, HTTP CRUD, WebServer UUID, or Custom), and how many waves to run (1–10).
2. **Wave 0 (Baseline)** — AutoBench generates the simplest correct implementation and benchmarks it. No tricks, no optimizations.
3. **Optimization Waves** — For each wave, AutoBench analyzes the code, proposes 3–5 optimizations across all layers (code, architecture, infra), and lets you pick which ones to apply. It implements them, runs the benchmark 3 times, averages the results, and compares against the baseline and previous wave.
4. **Rollback** — If a wave makes things worse, AutoBench records the findings honestly and lets you rollback. The next wave builds on the best-known version.
5. **Final Report** — After all waves, AutoBench generates a summary with comparison tables, an ASCII performance chart, and impact rankings for every optimization tried.

Anti-cheat rules are enforced: no hardcoded results, no skipping work, no pre-computed data. Every number comes from actual benchmark runs.

## Skill in action on CC

/autobench <br/>
<img src="autobench-in-action.png" width="600" />

Optimizations in Waves
<img src="optimizations.png" width="600" />

## Results — Java 25 CSV Analytics (1M rows, 32.6MB)

| Wave | Avg Time | Throughput | Rows/sec | vs Baseline |
|---|---|---|---|---|
| W0 (baseline) | 411.7ms | 79.2 MB/s | 2,428,953 | — |
| W1 | 308.6ms | 105.6 MB/s | 3,240,441 | -25.0% |
| W2 | 196.7ms | 165.7 MB/s | 5,083,884 | -52.2% |
| W3 (partial rollback) | 189.7ms | 171.9 MB/s | 5,271,481 | -53.9% |
| **W4 (BEST)** | **151.3ms** | **215.5 MB/s** | **6,609,385** | **-63.3%** |
| W5 (rolled back) | 208.6ms | 156.3 MB/s | 4,793,864 | rolled back |

**Final: 2.72x faster** — from 411.7ms to 151.3ms.

Top wins: mmap + multithreading + bulk byte copy + fixed-point arithmetic.

### Detailed Files

- [findings.md](findings.md) — wave-by-wave detailed findings
- [report.md](report.md) — final summary with impact rankings
- [bench.sh](bench.sh) — reproducible benchmark runner
