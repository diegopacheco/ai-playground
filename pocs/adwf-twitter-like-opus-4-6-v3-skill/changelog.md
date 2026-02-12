# Changelog

## 2026-02-12 - Initial Release

### Features
- User registration with username, email, and password (Argon2 hashing)
- User login with JWT authentication (24h expiry)
- Create posts (280 character limit)
- Delete own posts
- Like and unlike posts
- Follow and unfollow users (self-follow prevented)
- Timeline feed showing posts from followed users and own posts
- User profile pages with follower/following counts
- Single post detail view

### Backend (Rust + Axum)
- 13 API endpoints: register, login, me, create/get/delete posts, timeline, user posts, like/unlike, follow/unfollow, user profile, followers, following
- SQLite database via SQLx with 4 tables (users, posts, likes, follows)
- JWT middleware for protected routes
- CORS configuration for frontend origin
- UUID primary keys with composite keys on junction tables

### Frontend (React 19 + TypeScript)
- 5 pages: Home, Login, Register, Profile, Post Detail
- TanStack Router with auth-guarded routes
- TanStack Query with optimistic updates for likes/follows
- Tailwind CSS styling
- PostComposer with live character counter
- Responsive layout

### Database
- SQLite schema with TEXT primary keys (UUIDs)
- Composite primary keys on likes and follows tables
- Indexes on user_id, created_at, post_id, following_id
- Foreign key constraints defined

### Tests
- 13 Rust unit tests (JWT, error mapping, model conversion)
- 20 Rust integration tests (all API endpoints)
- 8 Playwright UI tests (register, login, post, like, navigation)
- K6 stress test (4030 checks, 0% failure, p95=269ms, 20 VUs)

### Files Created
- `backend/` - Rust/Axum API server (11 source files)
- `frontend/` - React/TypeScript/Vite frontend (18 source files)
- `db/` - SQLite schema and shell scripts
- `backend/tests/integration_test.rs` - Integration test suite
- `frontend/tests/app.spec.ts` - Playwright UI tests
- `k6/stress-test.js` - K6 stress test
- `review/2026-02-12/` - Code review, security review, features, summary
- `design-doc.md` - Architecture and design documentation

### Review Findings
- 1 critical: hardcoded JWT secret (documented, not fixed)
- 2 high: no rate limiting, no password strength requirements
- 4 medium: permissive CORS, no token refresh, FK not enforced at runtime, CSRF considerations
- 3 low: email exposure in profiles, user enumeration, localStorage token storage
