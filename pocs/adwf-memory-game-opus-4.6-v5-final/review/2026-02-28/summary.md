# Memory Game - Changes Summary

**Date:** 2026-02-28

---

## 1. What Was Built

A full-stack memory card game where players flip cards to find matching pairs, earn scores, and compete on a leaderboard.

- **Backend (Rust/Axum):** REST API with endpoints for player management, game creation, card flipping, player stats, and a leaderboard. Game state is server-authoritative with card values hidden from clients until flipped.
- **Frontend (React/TypeScript):** Single-page application using TanStack Query for server state, Tailwind CSS for styling, and components for the game board, player registration, scoreboard, game-over overlay, and leaderboard.
- **Database (SQLite):** Three tables (`players`, `games`, `flips`) storing player identities, game state (board as JSON blob), move counts, scores, and flip history.

---

## 2. Files Created

### Backend (Rust)
| File | Purpose |
|------|---------|
| `backend/Cargo.toml` | Rust dependencies (axum 0.8, sqlx 0.8, tokio, serde, tower-http, rand) |
| `backend/Cargo.lock` | Dependency lock file |
| `backend/src/main.rs` | Server entry point, DB directory setup, listener binding |
| `backend/src/lib.rs` | Router setup, CORS configuration, route definitions |
| `backend/src/handlers.rs` | Request handlers for players, games, flips, stats, leaderboard + 18 unit tests |
| `backend/src/models.rs` | Request/response structs with serde derive |
| `backend/src/db.rs` | SQLite pool creation and schema initialization |
| `backend/tests/integration_test.rs` | Integration test file |
| `backend/run.sh` | Script to start the backend server |

### Frontend (React/TypeScript)
| File | Purpose |
|------|---------|
| `frontend/src/App.tsx` | Root component, game flow orchestration |
| `frontend/src/main.tsx` | React DOM entry point |
| `frontend/src/api.ts` | API client functions (createPlayer, createGame, getGame, flipCard, getLeaderboard) |
| `frontend/src/types.ts` | TypeScript type definitions (Game, CardData, LeaderboardEntry) |
| `frontend/src/index.css` | Global styles and Tailwind imports |
| `frontend/src/components/Card.tsx` | Individual card rendering with flip/match states |
| `frontend/src/components/GameBoard.tsx` | 4x4 card grid, flip logic, TanStack Query mutations |
| `frontend/src/components/PlayerForm.tsx` | Player name input and registration |
| `frontend/src/components/ScoreBoard.tsx` | Live score, moves, and timer display |
| `frontend/src/components/GameOver.tsx` | Completion overlay with final stats |
| `frontend/src/components/Leaderboard.tsx` | Top scores table |
| `frontend/index.html` | HTML shell |
| `frontend/package.json` | NPM dependencies and scripts |
| `frontend/vite.config.ts` | Vite build configuration |
| `frontend/tsconfig.json` | TypeScript base config |
| `frontend/tsconfig.app.json` | App-specific TypeScript config |
| `frontend/tsconfig.node.json` | Node-specific TypeScript config |
| `frontend/eslint.config.js` | ESLint configuration |
| `frontend/run.sh` | Script to start the frontend dev server |

### Database
| File | Purpose |
|------|---------|
| `db/schema.sql` | SQL schema with tables, indexes, and foreign key pragma |
| `db/create-schema.sh` | Schema creation script |
| `db/start-db.sh` | Database start script |
| `db/stop-db.sh` | Database stop script |
| `db/run-sql-client.sh` | SQLite client launcher |
| `db/memory_game.db` | SQLite database file |

### Testing
| File | Purpose |
|------|---------|
| `frontend/src/components/Card.test.tsx` | 8 unit tests for Card component (Vitest) |
| `frontend/src/game-logic.test.ts` | 11 unit tests for game logic functions (Vitest) |
| `frontend/src/test-setup.ts` | Vitest test setup |
| `frontend/e2e/memory-game.spec.ts` | End-to-end tests (Playwright) |
| `frontend/playwright.config.ts` | Playwright configuration |
| `k6/stress-test.js` | K6 load/stress test script |
| `k6/run-stress-test.sh` | Script to execute stress tests |

### Project
| File | Purpose |
|------|---------|
| `design-doc.md` | Design document |
| `todo.md` | Task tracking |

---

## 3. Test Coverage

### Unit Tests - Backend (Rust)
- **18 tests** in `handlers.rs` covering board generation, score calculation, card serialization, flip logic, and game completion detection
- Tests are in-memory logic tests only; no HTTP handler or database layer tests
- No tests for error cases (invalid position, already matched card, completed game)

### Unit Tests - Frontend (Vitest)
- **Card component:** 8 tests covering rendering states, click behavior, disabled state, CSS classes
- **Game logic:** 11 tests for `isGameComplete`, `getFlippedCards`, `calculateScore`, `createBoard`
- **Gap:** Game logic tests validate functions defined locally in the test file, not imported from application code
- **Gap:** No tests for PlayerForm, GameBoard, ScoreBoard, GameOver, Leaderboard, or api.ts

### Integration Tests
- Integration test file exists (`backend/tests/integration_test.rs`) but no HTTP-level handler tests with a real database

### E2E Tests (Playwright)
- End-to-end test spec at `frontend/e2e/memory-game.spec.ts`
- Playwright configuration at `frontend/playwright.config.ts`

### Stress Tests (K6)
- Load testing script at `k6/stress-test.js`
- Runner script at `k6/run-stress-test.sh`

---

## 4. Review Findings Summary

### Code Review - Critical Issues

| ID | Issue | Status |
|----|-------|--------|
| C1 | Frontend/backend type mismatch: field names differ (`is_flipped`/`is_matched` vs `flipped`/`matched`) and value types differ (`string` vs `Option<i32>`) | Open - application will not render correctly at runtime |
| C2 | `flipCard` API returns `Promise<Game>` but backend sends `FlipResponse { game, matched }` wrapper | Open - game board will not update after flipping |
| C3 | Race condition in `flip_card` handler: read-modify-write without transaction or locking | Open - concurrent flips can corrupt game state |

### Code Review - Major Issues

| ID | Issue |
|----|-------|
| M1 | Pervasive `.unwrap()` usage across handlers, db, and main -- any DB error crashes the server |
| M2 | Schema divergence between `db.rs` and `schema.sql` (indexes, NOT NULL, foreign keys differ) |
| M3 | Foreign keys not enforced (missing `PRAGMA foreign_keys = ON` in `db.rs`) |
| M4 | `LeaderboardEntry` type has `time_seconds` field that does not exist in backend response |
| M5 | Returning players blocked by 409 UNIQUE constraint on player name |

### Security Review - Key Findings

| Severity | ID | Finding |
|----------|----|---------|
| High | H-1 | Race condition in flip_card (no transaction isolation) |
| Medium | M-1 | No input length validation on player names |
| Medium | M-2 | No rate limiting on any endpoint |
| Medium | M-3 | No authentication or authorization |
| Medium | M-4 | Foreign keys not enforced |
| Medium | M-5 | Card values revealed on flip enable automated perfect play |
| Low | L-1 | Sequential predictable resource IDs |
| Low | L-2 | Hardcoded CORS origin |
| Low | L-3 | Unused `@tanstack/react-router` dependency |
| Low | L-4 | Hardcoded database path |
| Low | L-5 | `.unwrap()` on DB operations causes panics instead of HTTP 500 |

### Positive Findings
- No SQL injection vulnerabilities (all queries use parameterized bindings)
- No XSS vulnerabilities (React default escaping, no `dangerouslySetInnerHTML`)
- Card values correctly hidden from API when cards are face-down
- CORS is restricted (not wildcard), methods and headers are appropriately scoped
- All dependencies are current versions

### Remaining Recommendations (Priority Order)

1. Fix frontend/backend type mismatches (C1, C2, M4) -- application is broken at runtime
2. Wrap `flip_card` in `BEGIN EXCLUSIVE` transaction to prevent race conditions
3. Replace `.unwrap()` with proper error handling returning HTTP 500
4. Enable `PRAGMA foreign_keys = ON` and consolidate schema definitions
5. Add input validation (player name length limits)
6. Add rate limiting middleware
7. Make CORS origin and database path configurable via environment variables
8. Add "login with existing name" flow for returning players
9. Remove unused `@tanstack/react-router` dependency
10. Extract game logic into testable pure functions and add integration tests

---

## 5. Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Backend Runtime | Rust | stable |
| Backend Framework | Axum | 0.8 |
| Async Runtime | Tokio | 1 |
| Database Driver | SQLx | 0.8 (SQLite) |
| HTTP Middleware | tower-http | 0.6 |
| Serialization | serde / serde_json | 1 |
| Randomization | rand | 0.9 |
| Database | SQLite | embedded |
| Frontend Framework | React | 19.2.0 |
| Frontend Language | TypeScript | via Vite |
| Build Tool | Vite | latest |
| CSS Framework | Tailwind CSS | 4.2.1 |
| Server State | TanStack Query | 5.90.21 |
| Unit Testing (FE) | Vitest | latest |
| E2E Testing | Playwright | latest |
| Stress Testing | K6 | latest |
| Package Manager | Bun | latest |
