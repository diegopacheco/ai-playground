# Code Review - 2026-02-13

## Overall Assessment

The codebase is well-structured and follows idiomatic patterns for both Rust/Axum and React/TypeScript. The separation of concerns is clean with distinct layers for handlers, models, middleware, and database access.

## Backend (Rust/Axum)

### Strengths
- Clean route separation between protected and public routes
- Proper use of Axum extractors (State, Extension, Path, Json)
- JWT auth middleware correctly validates tokens and injects AuthUser into request extensions
- BCrypt password hashing with cost factor 10
- Parameterized SQL queries throughout (no SQL injection risk)
- Proper HTTP status codes returned (201 for create, 204 for delete, 401/404/409 as appropriate)
- Tweet delete verifies ownership (user_id check in WHERE clause)
- Feed query includes own tweets plus followed users' tweets
- mod.rs files only re-export modules (no logic)

### Issues Found
- `create_tweet` in tweet_handler.rs has a fallback query pattern where the first query uses a subquery syntax that may fail, then falls back to a CTE-based query. The first query should be removed since the CTE approach is correct.
- `db.rs` uses inline SQL for migrations rather than the schema.sql file. This duplicates the schema definition. Consider using `schema.sql` or sqlx migrations.
- CORS allows `Any` for methods and headers which is permissive. Consider restricting to only needed methods (GET, POST, PUT, DELETE) and headers (Content-Type, Authorization).
- No pagination on get_user_tweets endpoint (feed is limited to 50 but user tweets has no limit).
- No rate limiting on auth endpoints (login/register).

### Suggestions
- Add pagination parameters (offset/limit) to list endpoints
- Consider using sqlx migrations directory instead of inline SQL
- Add request body size limits

## Frontend (React/TypeScript)

### Strengths
- TypeScript throughout with proper type definitions
- TanStack Query for server state with appropriate stale time (30s)
- Auth context with localStorage persistence
- Axios interceptor for automatic JWT header injection
- Protected routes redirect to login
- Character counter on compose box with 280 limit
- Clean component hierarchy: App > Pages > Components

### Issues Found
- JWT token stored in localStorage (vulnerable to XSS, but acceptable for this scope)
- No token expiration check on the frontend - expired tokens will cause 401s
- No error boundary component wrapping the app
- QueryClient is created outside the component tree (correct pattern)

### Suggestions
- Add loading states/spinners for data fetching
- Add toast notifications for actions (tweet created, followed, etc.)

## Database

### Strengths
- Proper foreign key constraints
- ON DELETE CASCADE on likes when tweet is deleted
- Self-follow prevention via CHECK constraint
- Appropriate indexes on query columns
- Seed data uses ON CONFLICT DO NOTHING for idempotency

## Test Coverage

- Backend: 16 unit tests covering auth, config, and tweet validation
- Frontend: 25 unit tests covering ComposeBox, TweetCard, NavBar, useAuth
- Integration, E2E, and stress test scripts created

## Verdict

Code quality is good. No critical bugs found. The application follows standard patterns and is well-organized.
