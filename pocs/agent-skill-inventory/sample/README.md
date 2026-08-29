# Sample report

A real `/inventory` run against
[`pocs/adwf-twitter-like`](../../adwf-twitter-like) — a Rust axum backend over
PostgreSQL with a React 19 single page app.

| File | What it is |
| --- | --- |
| [`index.html`](index.html) | The report. Open it in a browser; it needs nothing else. |
| [`facts.json`](facts.json) | What `scan.py` measured. No judgement, only evidence. |
| [`analysis.json`](analysis.json) | What survived all three passes. |

The report header still points at the temp directory it was generated in, which
is what the skill writes at the top of every report.

## What the run found

* 8 modules, 166 files, 9,801 lines, 6 PostgreSQL tables.
* 12 tech-debt items, 5 of them high severity: a feed that runs about 121
  queries per page, validator rules that no handler ever calls, a JWT secret
  that falls back to a literal string, tokens in `localStorage`, and CORS open
  to every origin.
* Logging scored 18 of 100 — the whole Rust backend writes three log lines.
* 29 test files and 256 cases across four kinds; nothing tests failure.
