# Changes Summary - 2026-02-13

## What Was Built

A full-stack Twitter-like social media application with user registration/login, tweet creation (280 char limit), likes, follows, and a home feed.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Rust 1.93+, Axum 0.8, Tokio, SQLx |
| Frontend | React 19, TypeScript, Vite 6, Tailwind CSS 4, TanStack Query |
| Database | PostgreSQL 18 (containerized via Podman) |
| Auth | JWT (jsonwebtoken crate) + BCrypt |
| Package Manager | Bun (frontend), Cargo (backend) |

## Files Created

### Backend (17 files)
- Cargo.toml + 16 Rust source files
- Handlers: auth, user, tweet, like, follow
- Models: user, tweet, like, follow
- Middleware: JWT auth
- Config, DB pool, JWT auth module

### Frontend (20 files)
- Package.json, vite config, TS configs
- 5 pages: Login, Register, Home, Profile, TweetDetail
- 5 components: NavBar, TweetCard, UserCard, ComposeBox, ProtectedRoute
- 4 API modules: client, auth, tweets, users
- Auth context + useAuth hook
- Type definitions

### Database (6 files)
- schema.sql with 4 tables + 5 indexes + seed data
- start-db.sh, stop-db.sh, create-schema.sh, run-sql-client.sh
- podman-compose.yml

### Tests (12 files)
- Backend: 16 unit tests (auth, config, tweet validation)
- Frontend: 25 unit tests (ComposeBox, TweetCard, NavBar, useAuth)
- Integration test script, Playwright e2e tests, K6 stress tests
- 4 test runner scripts

### Infrastructure (4 files)
- build.sh, run.sh
- design-doc.md, todo.md, mistakes.md

## Test Results

| Suite | Tests | Status |
|-------|-------|--------|
| Backend Unit | 16 | All Pass |
| Frontend Unit | 25 | All Pass |
| Integration | Script ready | Requires running services |
| E2E (Playwright) | Script ready | Requires running services |
| K6 Stress | Script ready | Requires running services |

## Key Architectural Decisions

1. **Axum over Actix-web**: Chosen for its integration with the Tower ecosystem and simpler middleware model
2. **SQLx over Diesel**: Async-native, no DSL overhead, direct SQL queries
3. **TanStack Query**: Automatic caching and invalidation for server state
4. **JWT in localStorage**: Simple auth approach suitable for SPA architecture
5. **Inline migrations in db.rs**: Schema also maintained in db/schema.sql for container setup
6. **Podman over Docker**: Per project requirements
