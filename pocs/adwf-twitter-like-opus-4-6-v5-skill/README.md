# Twitter-like Application

A full-stack Twitter-like social media application built with Rust (Axum), React 19, and PostgreSQL.

## Documentation

- [Design Document](design-doc.md)
- [Code Review](review/2026-02-13/code-review.md)
- [Security Review](review/2026-02-13/sec-review.md)
- [Feature Documentation](review/2026-02-13/features.md)
- [Changes Summary](review/2026-02-13/summary.md)
- [Changelog](changelog.md)

## Highlights

- 15 REST API endpoints with JWT authentication
- React 19 + TypeScript + TanStack Query frontend
- PostgreSQL with proper constraints, indexes, and seed data
- 41 unit tests passing (16 backend + 25 frontend)
- Integration, E2E (Playwright), and stress (K6) test scripts included

## Quick Start

### Prerequisites
- Rust 1.93+
- Bun
- Podman

### Database
```bash
cd db
./start-db.sh
./create-schema.sh
```

### Backend
```bash
cd backend
DATABASE_URL="postgresql://twitter:twitter123@localhost:5432/twitter" cargo run
```
Backend runs on http://localhost:8080

### Frontend
```bash
cd frontend
bun install
bun run dev
```
Frontend runs on http://localhost:5173

### All-in-One
```bash
./run.sh
```

### Default Credentials
- Email: admin@twitter.local
- Password: admin123

## Running Tests

```bash
./run-unit-tests.sh
./run-integration-tests.sh
./run-e2e-tests.sh
./run-stress-tests.sh
```

## Stopping

```bash
cd db
./stop-db.sh
```
