# Changelog

## 2026-02-28 - Memory Game v1.0.0

### Built
- Full-stack memory card matching game
- Rust/Axum REST API backend (port 3000) with 6 endpoints
- React 19/TypeScript frontend (Vite + Tailwind CSS + TanStack Query)
- SQLite database with 3 tables (players, games, flips)
- 4x4 card grid with 8 pairs, server-side game logic
- Scoring system: max(1000 - moves*10, 100)
- Player registration and leaderboard (top 10)
- 3D CSS card flip animations
- Client-side elapsed time tracker

### Files Created
- **Backend** (7 files): Cargo.toml, src/main.rs, src/lib.rs, src/db.rs, src/handlers.rs, src/models.rs, run.sh
- **Frontend** (16 files): package.json, vite.config.ts, index.html, src/main.tsx, src/App.tsx, src/api.ts, src/types.ts, src/index.css, 7 component files
- **Database** (6 files): schema.sql, memory_game.db, create-schema.sh, start-db.sh, stop-db.sh, run-sql-client.sh
- **Tests** (6 files): backend unit tests, integration_test.rs, Card.test.tsx, game-logic.test.ts, e2e/memory-game.spec.ts, k6/stress-test.js
- **Docs** (6 files): design-doc.md, todo.md, code-review.md, sec-review.md, features.md, summary.md

### Test Coverage
- 19 Rust unit tests (board generation, scoring, game state)
- 19 Rust integration tests (full API flow)
- 19 Vitest frontend tests (Card component, game logic)
- 5 Playwright e2e test specs
- K6 stress test script (10 VUs, 45s duration)

### Review Findings Fixed
- Fixed frontend/backend type mismatches (field names and types aligned)
- Fixed FlipCard API response handling (FlipResponse envelope)
- Added transaction safety to flip_card handler
- Added PRAGMA foreign_keys = ON at db initialization
- Replaced all .unwrap() calls with proper error handling

### Remaining Recommendations
- Add input length validation on player names
- Add rate limiting on API endpoints
- Add authentication/authorization for production use
- Align db/schema.sql with runtime schema in db.rs
