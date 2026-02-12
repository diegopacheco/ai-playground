# Twitter-Like Application

Full-stack Twitter clone built with Rust (Axum) backend, React 19 (TypeScript) frontend, and SQLite database.

## Overview

A social media application supporting user registration/login, posting (280 char limit), likes, follows, timeline feeds, and user profiles. JWT-based authentication protects all API endpoints.

### Stack
- **Backend**: Rust + Axum 0.8, SQLx, JWT, Argon2
- **Frontend**: React 19, TypeScript, Vite, TanStack Router + Query, Tailwind CSS
- **Database**: SQLite
- **Tests**: Cargo (unit + integration), Playwright (UI), K6 (stress)

## Quick Start

### Database
```bash
cd db
./create-schema.sh
```

### Backend
```bash
cd backend
cargo run
```
Runs on http://localhost:3000

### Frontend
```bash
cd frontend
bun install
bun run dev
```
Runs on http://localhost:5173

## Test Results

| Suite | Result |
|---|---|
| Unit Tests (Rust) | 13/13 passed |
| Integration Tests (Rust) | 20/20 passed |
| UI Tests (Playwright) | 8/8 passed |
| Stress Test (K6) | 4030 checks, 0% fail, p95=269ms |

### Running Tests
```bash
cd backend && cargo test
cd frontend && bunx playwright test
k6 run k6/stress-test.js
```

## Documentation

- [Design Doc](design-doc.md)
- [Code Review](review/2026-02-12/code-review.md)
- [Security Review](review/2026-02-12/sec-review.md)
- [Features](review/2026-02-12/features.md)
- [Changes Summary](review/2026-02-12/summary.md)
- [Changelog](changelog.md)

## Highlights

- 13 REST API endpoints with JWT auth middleware
- Optimistic UI updates for likes and follows
- Auth-guarded routing via TanStack Router beforeLoad hooks
- Composite primary keys on junction tables preventing duplicates at DB level
- Argon2 password hashing with random salt per user
- Full test coverage: unit, integration, UI, and stress tests all passing
