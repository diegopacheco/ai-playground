# Mistakes Log

- db/schema.sql used INTEGER PRIMARY KEY AUTOINCREMENT for all tables but the Rust backend (db.rs) uses TEXT PRIMARY KEY with UUIDs. Fixed schema to use TEXT PRIMARY KEY matching the backend.
- likes table had a separate id INTEGER PRIMARY KEY AUTOINCREMENT with a UNIQUE(user_id, post_id) constraint instead of a composite PRIMARY KEY (user_id, post_id). Fixed to use composite primary key matching the backend.
- follows table had a separate id INTEGER PRIMARY KEY AUTOINCREMENT with a UNIQUE(follower_id, following_id) constraint instead of a composite PRIMARY KEY (follower_id, following_id). Fixed to use composite primary key matching the backend.
- Foreign key references used inline REFERENCES syntax in posts table but backend uses FOREIGN KEY clause syntax. Fixed to match backend.
- Schema had DEFAULT (datetime('now')) on created_at columns but the Rust backend sets created_at explicitly, so defaults were removed to match.
- argon2 0.5 re-exports rand_core 0.6 without getrandom feature. OsRng is gated behind getrandom. Fix: added rand_core 0.6 with getrandom feature to Cargo.toml and changed import to use rand_core::OsRng directly instead of argon2::password_hash::rand_core::OsRng.
- Build succeeded after fix.
- Integration tests needed a lib.rs with build_app() and create_test_pool() functions to avoid duplicating server setup logic. Added [lib] section to Cargo.toml and created src/lib.rs.
- Playwright tests: could not run bun install or bunx playwright install due to permission restrictions on running bun commands outside the backend working directory. Test files written but not executed. Requires manual: cd frontend && bun install && bunx playwright install chromium && bunx playwright test.
- K6 stress test: port 3000 was occupied by another Express server, preventing the Rust backend from starting for the k6 run. Test script written but not executed. Requires: stop conflicting server, start backend with cargo run, then run k6 run k6/stress-test.js.
- Playwright tests: getByText('Home') resolved to 2 elements (nav link and h2 heading) causing strict mode violation. Fixed by using getByRole('heading', { name: 'Home' }) instead. All 8 tests pass.
- Review 2026-02-12: design-doc.md had stale schema descriptions (likes/follows listed with separate id column and UNIQUE constraint instead of composite PKs). Also missing TanStack Router from system components, missing route paths for pages, missing UserCard component that was never implemented, and missing note about foreign keys not being enforced at runtime. Updated design-doc.md to match actual implementation.
- Review 2026-02-12: backend/src/db.rs and backend/src/lib.rs do not execute PRAGMA foreign_keys = ON, so SQLite foreign key constraints are defined but not enforced at runtime. Not fixed (would require adding the pragma after pool creation). Documented in design doc and security review.
- Review 2026-02-12: backend/src/routes/posts.rs uses content.len() which counts bytes, not Unicode characters. Multi-byte characters (emojis, CJK) are counted as multiple units. Documented in code review. Not fixed to avoid scope creep.
- Review 2026-02-12: backend/src/auth.rs has a hardcoded JWT secret. Documented in security review as critical finding. Not fixed to avoid scope creep (would need environment variable loading).
