# Autobench in Action

## What is AutoBench?

AutoBench is a Claude Code skill that runs automated, iterative performance benchmarking on code. You pick a language, a benchmark type, and a number of optimization waves. AutoBench then generates a naive baseline implementation, measures it, and runs wave after wave of optimizations — each time proposing changes, letting you approve them, benchmarking the result, and honestly recording whether things got better or worse.

## Sample Bench

A full Java CSV Analytics benchmark run with 5 optimization waves is available in the [sample-bench/](sample-bench/) folder.

**Final: 2.72x faster** — from 411.7ms to 151.3ms.

### Key Files

- [sample-bench/README.md](sample-bench/README.md) — full results with screenshots and comparison table
- [sample-bench/findings.md](sample-bench/findings.md) — wave-by-wave detailed findings
- [sample-bench/report.md](sample-bench/report.md) — final summary with impact rankings
- [sample-bench/bench.sh](sample-bench/bench.sh) — reproducible benchmark runner
- [sample-bench/src/CsvAnalytics.java](sample-bench/src/CsvAnalytics.java) — optimized implementation
- [sample-bench/src/GenerateCSV.java](sample-bench/src/GenerateCSV.java) — CSV data generator
- [skills/autobench/SKILL.md](skills/autobench/SKILL.md) — the AutoBench skill definition
- [design-doc.md](design-doc.md) — design document
- [install.sh](install.sh) — install script
- [uninstall.sh](uninstall.sh) — uninstall script