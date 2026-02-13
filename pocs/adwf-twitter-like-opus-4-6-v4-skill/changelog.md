# Changelog

## 2026-02-12 - Initial Release

### Built
- Twitter-like social media application ("Chirp") with full-stack architecture
- Rust backend (Axum + SQLx + SQLite) with 16 REST API endpoints
- React 19 frontend (TypeScript + Vite + Bun + TanStack Query + Tailwind CSS)
- SQLite database with 4 tables, constraints, and 6 indexes

### Backend (Rust)
- User registration with bcrypt password hashing and JWT token response
- User login with JWT authentication (24-hour expiry)
- Post CRUD (create, read, delete) with 280-character limit
- Like/unlike system with duplicate prevention
- Follow/unfollow system with self-follow prevention
- Personalized feed (posts from followed users + own posts, paginated)
- CORS configured for frontend origin
- Proper error handling with custom AppError enum

### Frontend (React 19 + TypeScript)
- Login and registration pages with form validation
- Home page with post composer and infinite scroll feed
- Profile page with posts, followers, following tabs
- Post detail page with like interaction
- Navbar with navigation
- TanStack Query for server state management
- React Context for auth state (persisted to localStorage)
- Tailwind CSS styling

### Database
- `users` table with unique username/email constraints
- `posts` table with 280-char content limit
- `likes` table with unique user-post constraint
- `follows` table with unique follower-following constraint and self-follow check
- 6 indexes for query optimization

### Files Created
- 15 Rust source files in `backend/src/`
- 22 TypeScript/TSX files in `frontend/src/`
- 6 frontend unit test files
- 4 database scripts in `db/`
- Integration test suite (13 API tests)
- 3 Playwright e2e tests
- K6 stress test script
- 4 test runner scripts at project root
- `build.sh` for full project build

### Test Coverage
- 23 backend unit tests (auth, models, errors)
- 19 frontend unit tests (components, pages)
- 13 integration tests (full API flow)
- 3 Playwright e2e tests (login page, registration, auth flow)
- K6 stress tests (10 VUs, 94 iterations, p95 < 2s)

### Review
- Code review: `review/2026-02-12/code-review.md`
- Security review: `review/2026-02-12/sec-review.md`
- Feature documentation: `review/2026-02-12/features.md`
- Changes summary: `review/2026-02-12/summary.md`

### Known Issues
- JWT secret is hardcoded (should use environment variable in production)
- No rate limiting on API endpoints
- No pagination metadata in response (total count, has_more)
- Frontend uses simple state-based routing instead of proper router
