# Twitter Like Clone

## Overview
This project provides a Twitter-like web clone with a Rust backend API, a React frontend, and SQLite persistence.

## Links
- Design doc: `design-doc.md`
- Code review: `review/2026-02-11/code-review.md`
- Security review: `review/2026-02-11/sec-review.md`
- Feature document: `review/2026-02-11/features.md`
- Changes summary: `review/2026-02-11/summary.md`
- Changelog: `changelog.md`

## Summary Highlights
- Backend and frontend core feature flow is implemented and validated.
- Unit, integration, UI, and stress test executions completed.
- No critical blockers identified in review.

## Quick Start
1. Backend
- `cd backend`
- `cargo run`

2. Frontend
- `cd frontend`
- `bun run start`

3. Database
- Schema is auto-initialized by backend startup.
- Optional manual apply: `sqlite3 db/app.db < db/schema.sql`

## Tests
- Unit tests: `cd backend && cargo test`
- Integration tests: `./tests/integration.sh`
- UI tests: `./tests/ui.sh`
- Stress tests: `./tests/stress.sh`
