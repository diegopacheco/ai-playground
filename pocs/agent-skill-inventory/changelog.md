# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-08-29

### Added

* `/inventory` skill that scans a codebase three times and renders one
  self-contained light-theme searchable HTML report.
* `scripts/scan.py` — mechanical evidence collector. Detects modules from build
  files with a directory fallback, classifies test files into eleven kinds and
  counts their cases, counts log statements by level, finds SQL tables across
  raw DDL, Liquibase, JPA, Prisma and Django, flags risky queries, and collects
  tech-debt signals: markers, large files, long functions, credential-shaped
  literals and duplicate dependencies.
* `scripts/render.py` — validating renderer. Rejects an analysis with a path
  that does not exist, a module without exactly 5 pros and 5 cons, a count that
  disagrees with the scan, an edge to an unknown node, a severity outside
  high/medium/low, more than 10 tech-debt items in one module, or fewer than
  three passes.
* Hand-drawn architecture diagram generator: layered layout, side-routed long
  edges, label collision avoidance and a content-fitted viewBox, so nothing
  overlaps.
* Seven report tabs — architecture, modules, schema, committers, tech debt,
  observability, tests — with search across all of them at once and per-tab
  match badges.
* Committer pictures from GitHub when a login is known, Gravatar otherwise, and
  an inline initials SVG when both fail.
* Three externalized pass prompts in `prompts/` plus the `analysis.json`
  contract in `prompts/schema.md`.
* `install.sh` and `uninstall.sh` that ask whether to target Claude Code, Codex
  or both.
* `test.sh` with 44 checks over a generated fixture repository, including ten
  deliberately broken analyses that must each be rejected.
* `sample/` — a committed report from a real run against a Rust and React
  codebase, with its `facts.json` and `analysis.json`.

### Fixed during development

* Test classification matched `it(` inside words such as `split(`, marking
  ordinary source files as test files.
* Long-function detection never terminated for Python and Ruby, flagging every
  function; it is now indentation-aware for those languages.
* The SQL extractor required ten characters between `SELECT` and `FROM`, so
  `SELECT COUNT(*) FROM …` matched a later `FROM` and captured several
  statements as one query.
* Every primary-key lookup was flagged as unbounded; the flag now requires a
  missing `LIMIT` together with either no `WHERE` or an `ORDER BY`.
* `git log` counted commits for the whole enclosing repository when scanning a
  subdirectory, and author directories were not scoped to the scanned path.
* Rust test cases were counted twice by two overlapping attribute patterns.
* The renderer compared test file counts but not case counts, letting a stale
  case number through.
