# Memory Game - Code Review

**Date:** 2026-02-28
**Scope:** Full-stack review of backend (Rust/Axum), frontend (React/TypeScript), and database (SQLite)

---

## 1. Overall Architecture Assessment

The project follows a clean client-server architecture with a Rust/Axum REST API backend, a React/TypeScript frontend using TanStack Query for server state management, and SQLite for persistence. The separation of concerns is reasonable: models, handlers, and database initialization live in distinct modules on the backend, while the frontend splits UI into focused components.

**Strengths:**
- Simple, understandable structure with minimal dependencies
- Game state is server-authoritative, preventing client-side cheating
- Card values are hidden from the API response when cards are face-down

**Weaknesses:**
- The backend and frontend have type mismatches that will cause runtime failures
- No input validation or sanitization on player names
- Game logic is embedded directly in the handler function rather than extracted into a testable domain layer

---

## 2. Backend Code Quality (Rust)

### 2.1 Error Handling

**Critical:** Nearly every database operation uses `.unwrap()`, which will panic and crash the server on any database error. The only handler with proper error handling is `create_player`, which matches on the result.

Affected locations in `/private/tmp/claude-tmpo/backend/src/handlers.rs`:
- `create_game` (line 133): `.unwrap()` on INSERT
- `get_game` (line 170): `.unwrap()` on SELECT
- `flip_card` (lines 225, 337, 350, 358): multiple `.unwrap()` calls on SELECT, UPDATE, and INSERT
- `get_player_stats` (lines 47, 66, 74, 82, 90): five separate `.unwrap()` calls
- `get_leaderboard` (line 402): `.unwrap()` on SELECT

In `/private/tmp/claude-tmpo/backend/src/main.rs`:
- Lines 7, 15, 21, 23: `.unwrap()` on directory creation, pool connection, listener binding, and serve

In `/private/tmp/claude-tmpo/backend/src/db.rs`:
- Lines 13, 32, 45: `.unwrap()` on all three CREATE TABLE statements

**Recommendation:** Return proper HTTP 500 errors from handlers instead of panicking. Use the `?` operator with a custom error type or Axum's `IntoResponse` for error variants.

### 2.2 API Design

- The `flip_card` endpoint returns a `FlipResponse` wrapping a `GameResponse` plus a `matched` field, but the `flipCard` function in the frontend API layer (`/private/tmp/claude-tmpo/frontend/src/api.ts`, line 31) declares the return type as `Promise<Game>`, not `Promise<FlipResponse>`. The frontend will receive `{ game: {...}, matched: ... }` but try to read it as a flat `Game` object. This is a **critical runtime bug**.

- The CORS configuration in `/private/tmp/claude-tmpo/backend/src/lib.rs` (line 12) is hardcoded to `http://localhost:5173`. This will break in any non-local deployment.

- The `get_player_stats` handler makes 4 sequential database queries (lines 60-90) that could be consolidated into a single query with aggregation.

### 2.3 Game Logic Correctness

**Critical - Race Condition:** The `flip_card` handler in `/private/tmp/claude-tmpo/backend/src/handlers.rs` performs a read-modify-write cycle (read game state at line 219, modify in memory, write back at line 326) without any locking or transaction. Two concurrent flip requests for the same game can read the same state and produce inconsistent results. This is especially dangerous because the frontend sends flip requests without waiting for prior flips to fully resolve.

**Logic Issue - Unmatched Card Reset:** When a third card is flipped (lines 273-284), any previously flipped-but-unmatched cards are reset to face-down. However, the client-side timer in `GameBoard.tsx` (lines 34-38) uses a 1-second `setTimeout` to re-fetch the game state, but does not prevent the user from flipping a third card during that window on the server side. The server will silently reset the two previous cards and start fresh, which is valid game logic, but the frontend timing and server logic are not well-synchronized.

**Hardcoded Values:** The number of pairs is hardcoded to 8 in both `generate_shuffled_board()` (line 105) and the completion check (line 314). The `total_pairs` column exists in the database but is never used dynamically.

### 2.4 Performance Considerations

- The entire board (all 16 cards with their values) is stored as a JSON blob in the `board` TEXT column. Every flip operation deserializes the full board, modifies it, re-serializes, and writes it back. This works for 16 cards but the pattern does not scale.

- The `flips` table is written to on every flip (line 353) but is never read anywhere in the application. This is dead writes producing unused data.

- The two UPDATE queries in `flip_card` (lines 326-351) are nearly identical, differing only in the `completed_at` field. This duplication could be simplified.

---

## 3. Frontend Code Quality (React/TypeScript)

### 3.1 Component Design

Components are well-structured and focused. `Card`, `ScoreBoard`, `GameOver`, `PlayerForm`, `Leaderboard`, and `GameBoard` each handle a single responsibility. Props interfaces are defined inline, which is fine for this project size.

**Issue in Card.tsx:** The `Card` component (`/private/tmp/claude-tmpo/frontend/src/components/Card.tsx`, line 15) accepts an `index` prop but the `index` parameter is only used for the `data-testid` attribute. The actual card position comes from `card.position`. This creates confusion about which identifier is authoritative.

### 3.2 State Management

TanStack Query is used appropriately for server state. The `QueryClient` is created at module scope in `App.tsx`, which is correct.

**Issue - Dynamic Import:** In `App.tsx` (line 25), `handlePlayAgain` uses a dynamic `import("./api")` to get `createGame`. This is unnecessary since `createGame` could be imported statically at the top of the file. Dynamic imports add complexity for no benefit here.

**Issue - Missing Error Handling in handlePlayAgain:** The `handlePlayAgain` function (lines 22-29) calls `createGame` directly without try/catch. If the API call fails, the error will be unhandled and silently swallowed.

### 3.3 API Integration

**Critical - Type Mismatch Between Frontend and Backend:**

The frontend `CardData` type in `/private/tmp/claude-tmpo/frontend/src/types.ts` uses:
```
value: string
is_flipped: boolean
is_matched: boolean
```

The backend `CardResponse` in `/private/tmp/claude-tmpo/backend/src/models.rs` sends:
```
value: Option<i32>
flipped: bool
matched: bool
```

There are three mismatches:
1. `value` is `string` on the frontend but `Option<i32>` (serialized as `number | null`) on the backend
2. `is_flipped` vs `flipped` - different field names
3. `is_matched` vs `matched` - different field names

This means `card.is_flipped` and `card.is_matched` will always be `undefined` (falsy) on the frontend, and `card.value` will be a number, not a string. The `CARD_COLORS` map in `Card.tsx` uses string keys ("A", "B", etc.) but will receive integers (1-8), so the color lookup will always fall through to `"bg-gray-500"`.

**Additionally**, as noted in Section 2.2, the `flipCard` API function returns `Promise<Game>` but the backend sends a `FlipResponse` wrapper. The `onSuccess` callback in `GameBoard.tsx` (line 26) sets `updatedGame` directly as the game cache, but it is actually `{ game: GameResponse, matched: Option<bool> }`. Every field access on the "game" will be undefined.

**The `LeaderboardEntry` type** has a `time_seconds` field that does not exist in the backend response. The backend `LeaderboardEntry` struct only has `player_name`, `score`, and `moves`. The leaderboard table will show `undefined` in the Time column.

### 3.4 UI/UX Patterns

- The `ScoreBoard` timer (`/private/tmp/claude-tmpo/frontend/src/components/ScoreBoard.tsx`, lines 14-21) correctly cleans up the interval on unmount and stops when the game completes.

- The `GameOver` overlay uses a fixed position backdrop, which is a good pattern for modals.

- The "Play Again" flow re-creates a game for the same player, but the player name is submitted as a new player each time. If a returning player enters the same name, `createPlayer` will fail with a CONFLICT (409) because of the UNIQUE constraint. The `PlayerForm` shows the error message, but the UX does not provide a "login" flow for existing players. This means a returning player cannot play again from the home screen.

---

## 4. Database Schema

### 4.1 Schema Design

The schema in `/private/tmp/claude-tmpo/db/schema.sql` is reasonable for the scope.

**Issue - Schema Divergence:** The `db.rs` file in the backend defines the schema inline with `CREATE TABLE IF NOT EXISTS` statements (lines 4-45) rather than reading from `schema.sql`. The `schema.sql` file includes indexes (lines 29-31) and `PRAGMA foreign_keys = ON` (line 1), but these are absent from `db.rs`. The backend will never create these indexes, and foreign keys will not be enforced at runtime.

**Issue - Foreign Key Definitions Differ:** In `schema.sql`, the foreign key is defined inline as `player_id INTEGER REFERENCES players(id)`, while in `db.rs` it uses the `FOREIGN KEY (player_id) REFERENCES players(id)` clause syntax. Both are valid SQL, but having two divergent schema definitions is a maintenance problem.

**Missing Constraint:** The `status` column accepts any TEXT value. A CHECK constraint limiting it to `'in_progress'` or `'completed'` would prevent invalid states.

**Missing NOT NULL:** The `player_id` column in `games` is `NOT NULL` in `db.rs` but has no `NOT NULL` in `schema.sql`.

### 4.2 Indexing Strategy

The `schema.sql` file defines three useful indexes:
- `idx_games_player_id` on `games(player_id)` - supports player stats queries
- `idx_games_status` on `games(status)` - supports leaderboard filtering
- `idx_flips_game_id` on `flips(game_id)` - supports flip lookups

However, as noted above, these indexes are never created by the application because `db.rs` does not include them. The leaderboard query joins `games` and `players` and filters by `status = 'completed'` with `ORDER BY score DESC` -- a composite index on `(status, score)` would be more effective than the single-column `status` index.

---

## 5. Test Coverage Assessment

### Backend Tests (`/private/tmp/claude-tmpo/backend/src/handlers.rs`, lines 420-664)

- 18 unit tests covering board generation, score calculation, card serialization, flip logic, and game completion detection
- Tests are purely in-memory logic tests -- no integration tests for the HTTP handlers or database layer
- The `flip_card` handler contains the most complex logic but is not tested through the handler interface, only through duplicated inline logic in tests
- No tests for error cases (invalid position, already matched, game completed)

### Frontend Tests

**Card component tests** (`/private/tmp/claude-tmpo/frontend/src/components/Card.test.tsx`):
- 8 tests covering rendering states, click behavior, disabled state, and CSS class application
- Note: The test file uses `CardData` with `is_flipped`/`is_matched` field names, which do not match what the backend actually sends (`flipped`/`matched`). The tests pass in isolation but do not reflect real runtime behavior.
- The test renders `<Card>` without an `index` prop (line 15), but the component type requires it. This test likely has a TypeScript error or the prop was made optional.

**Game logic tests** (`/private/tmp/claude-tmpo/frontend/src/game-logic.test.ts`):
- 11 tests for pure game state functions (`isGameComplete`, `getFlippedCards`, `calculateScore`, `createBoard`)
- These functions are defined locally in the test file itself, not imported from the application. They test logic that exists only in the test, not the actual application code.

**Missing Test Coverage:**
- No tests for `PlayerForm`, `GameBoard`, `ScoreBoard`, `GameOver`, or `Leaderboard` components
- No tests for the API layer (`api.ts`)
- No integration or end-to-end tests
- No backend integration tests with a real database

---

## 6. Issues Found

### Critical

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| C1 | Frontend/Backend type mismatch | `types.ts` vs `models.rs` | Field names differ (`is_flipped` vs `flipped`, `is_matched` vs `matched`) and value types differ (`string` vs `Option<i32>`). The application will not function correctly at runtime. |
| C2 | FlipCard API response mismatch | `api.ts:31` vs `handlers.rs:386-391` | Frontend expects `Game`, backend returns `FlipResponse { game, matched }`. The game board will not update after flipping cards. |
| C3 | Race condition in flip_card | `handlers.rs:214-392` | Read-modify-write on game state without transaction or locking. Concurrent flips corrupt game state. |

### Major

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| M1 | Pervasive `.unwrap()` usage | `handlers.rs`, `db.rs`, `main.rs` | Any database error crashes the server. Handlers should return HTTP 500 instead of panicking. |
| M2 | Schema divergence | `db.rs` vs `schema.sql` | Two separate schema definitions with differences in foreign keys, NOT NULL constraints, and indexes. Indexes in `schema.sql` are never created by the app. |
| M3 | Foreign keys not enforced | `db.rs` | Missing `PRAGMA foreign_keys = ON`. SQLite does not enforce foreign keys by default. |
| M4 | LeaderboardEntry type mismatch | `types.ts:31` | `time_seconds` field does not exist in backend response. Leaderboard Time column shows `undefined`. |
| M5 | Returning player blocked | `PlayerForm.tsx` | A player with an existing name gets a 409 error. No way to resume with an existing player identity. |

### Minor

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| m1 | Hardcoded CORS origin | `lib.rs:12` | Only allows `http://localhost:5173`. Will fail in production. |
| m2 | Hardcoded database path | `main.rs:9` | Uses relative path `../db/memory_game.db`. Should be configurable via environment variable. |
| m3 | Unused `flips` table | `handlers.rs:353` | Data is written but never read. |
| m4 | Hardcoded pair count | `handlers.rs:105,314` | `total_pairs` column exists but board size is always 8 pairs. |
| m5 | Dynamic import in App.tsx | `App.tsx:25` | Unnecessary dynamic `import("./api")` where a static import suffices. |
| m6 | Sequential DB queries in stats | `handlers.rs:60-90` | Four separate queries that could be a single aggregation query. |
| m7 | Frontend tests test local functions | `game-logic.test.ts` | Test-local helper functions are tested instead of actual application code. |
| m8 | Tuple destructuring for DB rows | `handlers.rs:164,219,397` | Using positional tuple fields (e.g., `row.0`, `row.1`) is fragile. Named struct mapping with `sqlx::FromRow` would be safer. |
| m9 | No input length validation | `handlers.rs:12-35` | Player name has no length limit. A malicious client can send extremely long strings. |

---

## 7. Recommendations

**Priority 1 - Fix Runtime-Breaking Issues:**
1. Align frontend `CardData` fields with backend `CardResponse` fields: rename `is_flipped` to `flipped`, `is_matched` to `matched`, and change `value` from `string` to `number | null`.
2. Fix the `flipCard` API function to return `FlipResponse` (with nested `game` and `matched` fields) or change the backend to return a flat `Game` response.
3. Remove `time_seconds` from the frontend `LeaderboardEntry` type or add it to the backend response.

**Priority 2 - Stability:**
4. Replace all `.unwrap()` calls in handlers with proper error responses. Use a shared error type that implements `IntoResponse`.
5. Wrap the read-modify-write cycle in `flip_card` in a database transaction with `BEGIN IMMEDIATE` to prevent concurrent modification.
6. Add `PRAGMA foreign_keys = ON` to the database initialization in `db.rs`.
7. Move index creation from `schema.sql` into `db.rs` so they are actually applied.

**Priority 3 - Usability:**
8. Add a "login with existing name" flow or use upsert semantics for player creation so returning players can start new games.
9. Make CORS origins and the database path configurable via environment variables.

**Priority 4 - Code Quality:**
10. Extract game logic from `flip_card` handler into a pure function that takes the current board state and a position, and returns the new board state plus metadata. This makes the core logic unit-testable without a database.
11. Replace tuple-based query results with named structs using `#[derive(sqlx::FromRow)]`.
12. Add integration tests for the HTTP API using `axum::test` helpers with an in-memory SQLite database.
13. Update frontend tests to test actual application behavior rather than test-local helper functions.
