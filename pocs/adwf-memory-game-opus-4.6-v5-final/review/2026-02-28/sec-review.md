# Security Review - Memory Game Project

**Date:** 2026-02-28
**Reviewer:** Security Reviewer Agent
**Scope:** Full-stack memory card game (Rust/Axum backend, React/TypeScript frontend, SQLite database)

---

## 1. Input Validation

### API Request Validation

**Backend (`handlers.rs`, `models.rs`):**

- `CreatePlayerRequest` accepts a `name: String` with no length validation. A client can submit an empty string, a single character, or a multi-megabyte string. The only frontend guard is `name.trim()` being truthy, but this is trivially bypassed via direct API calls.
- `FlipRequest` accepts `position: i32`, which allows negative values. While the `as usize` cast on line 247 of `handlers.rs` would wrap a negative i32 into a very large usize that would fail the bounds check (`pos >= cards.len()`), this is accidental safety rather than intentional validation.
- `CreateGameRequest` accepts `player_id: i64` with no validation that the player actually exists. The INSERT will succeed due to SQLite not enforcing foreign keys by default (see Section 8), creating orphaned game records.

**Severity: Medium**

### Frontend Input Sanitization

- `PlayerForm.tsx` trims whitespace before submission, which is adequate for the name field.
- No maximum length enforcement on the input field. A user could paste extremely long text.
- The frontend does not sanitize or validate any data received from the API before rendering.

**Severity: Low**

---

## 2. SQL Injection Assessment

All database queries in `handlers.rs` and `db.rs` use **parameterized queries** via SQLx's `.bind()` method. This is the correct approach.

Specific queries reviewed:
- `INSERT INTO players (name) VALUES (?)` -- parameterized via `.bind(&req.name)`
- `SELECT id, name FROM players WHERE id = ?` -- parameterized via `.bind(id)`
- `INSERT INTO games (player_id, board) VALUES (?, ?)` -- parameterized
- `SELECT ... FROM games WHERE id = ?` -- parameterized
- `UPDATE games SET board = ?, moves = ?, ...` -- parameterized
- `INSERT INTO flips (game_id, position) VALUES (?, ?)` -- parameterized
- `SELECT ... FROM games g JOIN players p ...` -- no user input in query, static

**Result: No SQL injection vulnerabilities found. All queries are properly parameterized.**

---

## 3. XSS Assessment

**React Output Escaping:**

React escapes all values rendered via JSX expressions `{}` by default. The project renders:
- `entry.player_name` in `Leaderboard.tsx` line 37 -- escaped by React
- `card.value` in `Card.tsx` line 40 -- escaped by React
- Numeric values (score, moves, time) throughout -- escaped by React

No usage of `dangerouslySetInnerHTML` was found anywhere in the codebase.

No raw HTML string concatenation or DOM manipulation via `innerHTML` is present.

**Result: No XSS vulnerabilities found. React's default escaping is sufficient.**

---

## 4. CORS Configuration

**File: `lib.rs` lines 11-18**

```rust
let cors = CorsLayer::new()
    .allow_origin(AllowOrigin::exact(
        "http://localhost:5173".parse().unwrap(),
    ))
    .allow_methods(AllowMethods::list([Method::GET, Method::POST]))
    .allow_headers(AllowHeaders::list([
        HeaderName::from_static("content-type"),
    ]));
```

**Findings:**

- The CORS origin is locked to `http://localhost:5173` (the Vite dev server). This is appropriate for development but will need to be updated for production deployment. It is not a wildcard (`*`), which is good.
- Only GET and POST methods are allowed, which matches the API surface.
- Only `content-type` header is allowed, which is appropriate.
- `allow_credentials` is not set, defaulting to false. This is correct since there is no cookie/session-based auth.

**Severity: Low (hardcoded dev origin needs production configuration)**

---

## 5. API Security

### Rate Limiting

**No rate limiting is implemented anywhere in the backend.**

An attacker can:
- Create unlimited player accounts via `POST /api/players`
- Create unlimited games via `POST /api/games`
- Send unlimited flip requests via `POST /api/games/{id}/flip`
- Flood the leaderboard with fake entries

This enables denial-of-service through database bloat and CPU/memory exhaustion.

**Severity: Medium**

### Authentication / Authorization

**No authentication or authorization exists in the application.**

- Any client can create games for any `player_id` without proving they own that player account.
- Any client can flip cards in any game by knowing the game ID (sequential integers, easily guessable).
- Any client can view any player's stats by iterating over player IDs.
- There are no sessions, tokens, API keys, or any form of identity verification.

For a casual game this is a design choice, but it means any user can manipulate any other user's active games.

**Severity: Medium**

### Data Exposure

- The `GET /api/games/{id}` endpoint correctly hides card values for face-down cards (returns `value: None` when `!card.flipped && !card.matched`).
- The `POST /api/games/{id}/flip` endpoint also correctly filters card values in the response.
- Game IDs are sequential integers, making enumeration trivial. An attacker can poll all active games.
- Player IDs are sequential integers, enabling enumeration of all players and their stats.

**Severity: Low**

---

## 6. Dependency Security

### Backend (Cargo.toml)

| Dependency | Version | Assessment |
|---|---|---|
| axum | 0.8 | Current major version, well-maintained |
| tokio | 1 | Standard async runtime, well-maintained |
| serde/serde_json | 1 | Standard serialization, well-maintained |
| sqlx | 0.8 | Current version, well-maintained |
| tower-http | 0.6 | Current version for CORS middleware |
| rand | 0.9 | Current version for shuffling |

### Frontend (package.json)

| Dependency | Version | Assessment |
|---|---|---|
| react | ^19.2.0 | Current major version |
| react-dom | ^19.2.0 | Current major version |
| @tanstack/react-query | ^5.90.21 | Current version |
| @tanstack/react-router | ^1.163.3 | Listed but not used in the code |
| tailwindcss | ^4.2.1 | Current version |

**Findings:**
- `@tanstack/react-router` is declared as a dependency but is not imported or used anywhere in the codebase. Unused dependencies increase attack surface unnecessarily.
- No lock file was reviewed for transitive dependency vulnerabilities. Running `cargo audit` and `npm audit` is recommended.

**Severity: Low**

---

## 7. Data Integrity

### Game State Manipulation Prevention

**Critical Finding -- Card Values Exposed via Rapid Flip Requests:**

The `flip_card` handler in `handlers.rs` returns the card value in the response when a card is flipped. An attacker can:
1. Send a flip request for position 0, receive the card value.
2. Send a flip request for position 1, receive the card value.
3. If they do not match, the two cards remain flipped until the next flip request resets them.
4. Record all values by flipping each card once.
5. Then replay perfect matches using the recorded values.

This is an inherent limitation of the game design where the server reveals card values on flip. A determined attacker with a simple script can always achieve a perfect score. This is hard to prevent in a client-server memory game without a fundamentally different architecture (e.g., the server only confirming matches without revealing values).

**Severity: Medium (inherent to the architecture)**

### Race Conditions

**Critical Finding -- No Concurrent Access Protection on Game State:**

The `flip_card` handler performs a read-modify-write cycle on the game state:
1. `SELECT` the game row (line 219-225)
2. Modify the cards in memory (lines 246-321)
3. `UPDATE` the game row (lines 326-351)

There is no database-level locking (e.g., `SELECT ... FOR UPDATE` or SQLite's `BEGIN EXCLUSIVE TRANSACTION`). If two concurrent flip requests arrive for the same game:
- Both read the same board state
- Both modify their local copy
- The second UPDATE overwrites the first one's changes
- This can corrupt the game state (lost flips, incorrect match counts)

SQLite's default journal mode provides some protection since writes are serialized, but the read-then-write gap still allows race conditions at the application level.

**Severity: High**

### Foreign Key Enforcement

The `db.rs` file does NOT execute `PRAGMA foreign_keys = ON`. SQLite disables foreign key enforcement by default. This means:
- Games can reference non-existent `player_id` values
- Flips can reference non-existent `game_id` values
- The `schema.sql` file includes `PRAGMA foreign_keys = ON` but this is not used by the application (the app creates tables in `db.rs`, not from the SQL file)

**Severity: Medium**

---

## 8. Vulnerabilities Found

### Critical

None.

### High

| ID | Finding | Location | Description |
|---|---|---|---|
| H-1 | Race condition in flip_card | `handlers.rs:214-392` | Read-modify-write without transaction isolation can corrupt game state under concurrent requests |

### Medium

| ID | Finding | Location | Description |
|---|---|---|---|
| M-1 | No input length validation | `handlers.rs:12-35` | Player name has no length limit, enabling storage abuse |
| M-2 | No rate limiting | `lib.rs` | No rate limiting on any endpoint enables DoS via resource exhaustion |
| M-3 | No authentication | All endpoints | Any client can manipulate any game or create entries for any player |
| M-4 | Foreign keys not enforced | `db.rs` | Missing `PRAGMA foreign_keys = ON` allows orphaned records |
| M-5 | Game state cheat via scripting | `handlers.rs:360-373` | Card values revealed on flip enable automated perfect play |

### Low

| ID | Finding | Location | Description |
|---|---|---|---|
| L-1 | Sequential/predictable resource IDs | All endpoints | Game and player IDs are sequential integers, enabling enumeration |
| L-2 | Hardcoded CORS origin | `lib.rs:12-14` | CORS origin is hardcoded to localhost dev server |
| L-3 | Unused dependency | `frontend/package.json` | `@tanstack/react-router` is declared but unused |
| L-4 | Hardcoded database path | `main.rs:9` | Database URL is hardcoded rather than configurable via environment |
| L-5 | Unwrap on DB operations | `handlers.rs` (multiple) | `.unwrap()` on database results will panic and crash the server on any DB error instead of returning a 500 response |

---

## 9. Recommendations

### Immediate (High Priority)

1. **Wrap flip_card in a transaction.** Use `sqlx::Transaction` with `BEGIN EXCLUSIVE` to prevent race conditions on game state updates. This is the most impactful fix.

2. **Add input validation for player names.** Enforce a minimum length of 1 and maximum length of 50 characters at the handler level. Reject names that are only whitespace.

3. **Enable foreign key enforcement.** Add `sqlx::query("PRAGMA foreign_keys = ON").execute(pool).await.unwrap();` at the start of `init_db()`.

4. **Replace `.unwrap()` with proper error handling.** Return HTTP 500 with a generic error message instead of panicking the server. Panics under load can cascade into service unavailability.

### Short-Term (Medium Priority)

5. **Add rate limiting.** Use `tower::limit::RateLimitLayer` or a similar middleware to limit requests per IP. Suggested limits: 10 player creations per minute, 5 game creations per minute, 60 flip requests per minute.

6. **Validate player_id existence** before creating a game. Return 404 if the player does not exist.

7. **Make CORS origin configurable** via environment variable for production deployment.

8. **Make database path configurable** via environment variable.

9. **Remove unused `@tanstack/react-router` dependency** from `package.json`.

### Long-Term (Low Priority)

10. **Add basic session management** if multiplayer or competitive leaderboard integrity matters. A simple token-based approach would prevent one player from manipulating another's games.

11. **Use UUIDs instead of sequential IDs** for games and players to prevent enumeration.

12. **Run `cargo audit` and `npm audit`** as part of CI/CD to catch known vulnerabilities in dependencies.

---

## Summary

The application has a solid foundation with no critical vulnerabilities. SQL injection and XSS are properly mitigated through parameterized queries and React's default escaping. The most significant issue is the race condition in the flip_card handler (H-1), which can corrupt game state under concurrent access. The lack of input validation, rate limiting, and authentication are notable gaps but are proportional risks for a single-player casual game. If the leaderboard is intended to be competitive, authentication (M-3) and anti-cheat measures (M-5) become more important.
