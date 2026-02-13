# Chirp - Twitter-like App

A full-stack Twitter-like social media application built with Rust (Axum) and React 19.

## Stack

- **Backend**: Rust 1.93+, Axum 0.8, SQLx 0.8, SQLite, JWT auth, bcrypt
- **Frontend**: React 19, TypeScript, Vite, Bun, TanStack Query, Tailwind CSS
- **Database**: SQLite (file-based)

## Features

- User registration and login with JWT authentication (24h expiry)
- Post creation (280 character limit) and deletion
- Like and unlike posts
- Follow and unfollow users
- Personalized feed (posts from followed users + own posts)
- User profiles with posts, followers, and following tabs
- Infinite scroll pagination

## Quick Start

### Database

```bash
cd db && sqlite3 ../backend/twitter.db < schema.sql
```

### Backend

```bash
cd backend
cargo build
cargo run
```

Backend runs on `http://localhost:8080`.

### Frontend

```bash
cd frontend
bun install
bun run dev
```

Frontend runs on `http://localhost:5173`.

### Full Build

```bash
./build.sh
```

## Running Tests

```bash
./run-unit-tests.sh
./run-integration-tests.sh
./run-e2e-tests.sh
./run-stress-tests.sh
```

| Category | Count | Tool |
|----------|-------|------|
| Backend unit tests | 23 | cargo test |
| Frontend unit tests | 19 | vitest |
| Integration tests | 13 | curl |
| E2E tests | 3 | Playwright |
| Stress tests | 10 VUs | K6 |

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | /api/users | No | Register |
| POST | /api/users/login | No | Login |
| GET | /api/users/me | Yes | Current user |
| GET | /api/users/{id} | No | User profile |
| GET | /api/users/{id}/followers | No | List followers |
| GET | /api/users/{id}/following | No | List following |
| POST | /api/users/{id}/follow | Yes | Follow user |
| DELETE | /api/users/{id}/follow | Yes | Unfollow user |
| POST | /api/posts | Yes | Create post |
| GET | /api/posts | No | List posts |
| GET | /api/posts/{id} | No | Get post |
| DELETE | /api/posts/{id} | Yes | Delete post |
| POST | /api/posts/{id}/like | Yes | Like post |
| DELETE | /api/posts/{id}/like | Yes | Unlike post |
| GET | /api/posts/{id}/likes | No | Like count |
| GET | /api/feed | Yes | Personal feed |

## Documentation

- [Design Doc](design-doc.md)
- [Changelog](changelog.md)
- [Code Review](review/2026-02-12/code-review.md)
- [Security Review](review/2026-02-12/sec-review.md)
- [Features](review/2026-02-12/features.md)
- [Changes Summary](review/2026-02-12/summary.md)

## Known Issues

- JWT secret is hardcoded (should use environment variable in production)
- No rate limiting on API endpoints
- Frontend uses state-based routing (no URL support)
- getUserPosts shows all posts instead of filtering by user
- liked_by_me field not populated by backend
