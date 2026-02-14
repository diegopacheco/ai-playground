# Changelog

## 2026-02-13 - Twitter-like Application (Initial Build)

### Built
- Full-stack Twitter-like application with Rust backend, React frontend, PostgreSQL database
- User authentication (register/login) with JWT tokens and BCrypt password hashing
- Tweet CRUD operations with 280-character limit
- Like/unlike tweets
- Follow/unfollow users
- Home feed showing tweets from followed users and own tweets
- User profile pages with follower/following counts

### Backend (Rust/Axum)
- 15 REST API endpoints across auth, users, tweets, likes, and follows
- JWT middleware for protected routes
- SQLx async PostgreSQL connection pool
- Zero clippy warnings, zero build errors

### Frontend (React 19/TypeScript)
- 5 pages: Login, Register, Home, Profile, TweetDetail
- 5 shared components: NavBar, TweetCard, UserCard, ComposeBox, ProtectedRoute
- TanStack Query for server state management
- Tailwind CSS styling
- Zero TypeScript errors, zero build warnings

### Database (PostgreSQL 18)
- 4 tables: users, tweets, likes, follows
- 5 indexes for query optimization
- Foreign keys, unique constraints, check constraints
- Seed admin user
- Podman container scripts

### Tests
- 16 backend unit tests (auth, config, tweet validation) - all passing
- 25 frontend unit tests (ComposeBox, TweetCard, NavBar, useAuth) - all passing
- Integration test script (curl-based API testing)
- Playwright E2E test scripts
- K6 stress test scripts

### Review
- Code review: no critical issues, well-structured codebase
- Security review: no critical vulnerabilities, recommendations for rate limiting and CORS tightening
- Feature documentation, design doc sync, changes summary

### Files Created/Modified
- 17 backend Rust source files
- 20 frontend TypeScript/TSX files
- 6 database files (schema, scripts, podman-compose)
- 12 test files and scripts
- 4 infrastructure scripts (build.sh, run.sh, test runners)
- 5 review documents
- design-doc.md, todo.md, mistakes.md, changelog.md, README.md
