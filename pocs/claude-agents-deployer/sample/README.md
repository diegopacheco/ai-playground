# Blog Platform

## Overview
This project is a Rust + React blog platform with admin controls for global comment availability and background theme selection.

## Documentation Links
- Design Doc: `design-doc.md`
- Code Review: `review/2026-02-09/code-review.md`
- Security Review: `review/2026-02-09/sec-review.md`
- Feature Documentation: `review/2026-02-09/features.md`
- Changes Summary: `review/2026-02-09/summary.md`
- Changelog: `changelog.md`

## Summary Highlights
- Admin panel at `/admin` controls comments and theme.
- Server blocks new comment creation when comments are disabled.
- Three supported background themes: `classic`, `forest`, `sunset`.

## Quick Start
### Database
- `./db/start-db.sh`
- `./db/create-schema.sh`

### Backend
- `cd backend`
- `cargo run`

### Frontend
- `cd frontend`
- `npm install`
- `npm run dev`

## Test Commands
- Backend unit tests: `cd backend && cargo test --lib`
- Backend integration tests: `cd backend && cargo test --test api_integration_test`
- Frontend build: `cd frontend && npm run build`
- K6 suite: `cd k6 && ./run-tests.sh`
